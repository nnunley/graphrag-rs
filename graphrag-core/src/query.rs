//! Total query front-end: arbitrary user text -> typed [`UserQueryProgram`].
//!
//! This is the ONLY place raw query strings are interpreted. The output is a
//! typed AST handed to leit's `Planner::plan_program`, so leit's textual
//! parser is never in the request path and operator punctuation inside prose
//! can never cause a parse failure.
//!
//! Interpretation rules:
//! - plain prose -> OR of terms (brain-style recall default)
//! - `"quoted words"` -> phrase node
//! - token-level uppercase `OR` / `AND` / `NOT` and balanced parens -> boolean nodes
//! - `field:value` -> fielded term ONLY when `field` is in the caller-supplied
//!   allowlist; unknown qualifiers and stray colons demote to plain terms
//!
//! Totality: the structured parser handles well-formed operator syntax; any
//! structural failure (unbalanced parens, dangling operators, unterminated
//! quotes, or nesting deeper than [`MAX_PARSE_DEPTH`]) falls back to an OR of
//! sanitized terms. `parse` therefore never
//! returns an error — `None` only means the input contained no searchable
//! tokens at all.

use std::sync::Arc;

use leit_index::{QueryBuilder, UserQueryProgram};

/// Parse arbitrary user text into a typed query program.
///
/// `allowed_fields` is the set of field names honored as `field:value`
/// qualifiers; any other qualifier is demoted to plain terms.
///
/// Returns `None` only when the input contains no searchable tokens (empty,
/// whitespace, or pure punctuation) — never for malformed operator syntax.
pub fn parse(query: &str, allowed_fields: &[&str]) -> Option<UserQueryProgram> {
    let tokens = tokenize(query)?;
    let mut builder = QueryBuilder::new();
    let mut pos = 0_usize;
    match parse_or(&tokens, &mut pos, &mut builder, allowed_fields, 0) {
        Ok(Some(root)) if pos == tokens.len() => {
            builder.set_root(root);
            builder.build()
        }
        // Structural failure, trailing garbage, or no operands: fall back to
        // an OR of sanitized plain terms so the query stays total.
        _ => fallback_terms(query),
    }
}

#[derive(Clone, Debug, PartialEq)]
enum Token {
    LParen,
    RParen,
    Word(String),
    Phrase(Vec<String>),
}

/// Tokenize into words, parens, and quoted phrases.
///
/// Returns `None` for input with no tokens. An unterminated quote is a
/// structural failure signalled by `None` here only when nothing else was
/// scanned; otherwise the caller's fallback path handles it (we surface it as
/// a dangling token sequence that fails the structured parse).
fn tokenize(query: &str) -> Option<Vec<Token>> {
    let mut tokens = Vec::new();
    let mut word = String::new();
    let mut chars = query.chars().peekable();
    let mut unterminated_quote = false;

    let flush = |word: &mut String, tokens: &mut Vec<Token>| {
        if !word.is_empty() {
            tokens.push(Token::Word(std::mem::take(word)));
        }
    };

    while let Some(c) = chars.next() {
        match c {
            '(' => {
                flush(&mut word, &mut tokens);
                tokens.push(Token::LParen);
            }
            ')' => {
                flush(&mut word, &mut tokens);
                tokens.push(Token::RParen);
            }
            '"' => {
                flush(&mut word, &mut tokens);
                let mut inner = String::new();
                let mut closed = false;
                for q in chars.by_ref() {
                    if q == '"' {
                        closed = true;
                        break;
                    }
                    inner.push(q);
                }
                if !closed {
                    unterminated_quote = true;
                }
                let terms: Vec<String> = inner
                    .split_whitespace()
                    .map(ToString::to_string)
                    .collect();
                if closed && !terms.is_empty() {
                    tokens.push(Token::Phrase(terms));
                } else {
                    // Unterminated or empty quote: contents become plain words.
                    for term in terms {
                        tokens.push(Token::Word(term));
                    }
                }
            }
            c if c.is_whitespace() => flush(&mut word, &mut tokens),
            c => word.push(c),
        }
    }
    flush(&mut word, &mut tokens);

    if unterminated_quote {
        // Force the structured parse to fail so the sanitized fallback runs.
        tokens.push(Token::RParen);
    }

    if tokens.is_empty() { None } else { Some(tokens) }
}

/// Structured-parse failure. Carries no detail: any failure routes to the
/// sanitized fallback.
struct Fail;

type ParseResult = Result<Option<leit_core::QueryNodeId>, Fail>;

/// Maximum operator-nesting depth (parens + NOT chains) the structured
/// parser will recurse into. Deeper input is a structural failure that
/// demotes to the sanitized fallback, keeping `parse` total: the bound keeps
/// recursion far below any realistic stack limit while being far deeper than
/// any human-written query.
const MAX_PARSE_DEPTH: usize = 64;

fn is_operand_start(token: &Token) -> bool {
    match token {
        Token::LParen | Token::Phrase(_) => true,
        Token::Word(w) => w != "OR" && w != "AND",
        Token::RParen => false,
    }
}

/// expr := and_expr ( ("OR" | adjacency) and_expr )*
///
/// Adjacency (two operands with no operator between them) means OR — the
/// brain-style prose default.
fn parse_or(
    tokens: &[Token],
    pos: &mut usize,
    builder: &mut QueryBuilder,
    allowed_fields: &[&str],
    depth: usize,
) -> ParseResult {
    if depth > MAX_PARSE_DEPTH {
        return Err(Fail); // too deep: demote to sanitized fallback
    }
    let mut children = Vec::new();
    loop {
        if let Some(child) = parse_and(tokens, pos, builder, allowed_fields, depth)? {
            children.push(child);
        }
        match tokens.get(*pos) {
            Some(Token::Word(w)) if w == "OR" => {
                *pos += 1;
                if !tokens.get(*pos).is_some_and(is_operand_start) {
                    return Err(Fail); // dangling OR
                }
            }
            Some(token) if is_operand_start(token) => {} // adjacency -> OR
            _ => break,
        }
    }
    Ok(match children.len() {
        0 => None,
        1 => Some(children[0]),
        _ => Some(builder.or(children)),
    })
}

/// and_expr := unary ( "AND" unary )*
fn parse_and(
    tokens: &[Token],
    pos: &mut usize,
    builder: &mut QueryBuilder,
    allowed_fields: &[&str],
    depth: usize,
) -> ParseResult {
    let mut children = Vec::new();
    loop {
        if let Some(child) = parse_unary(tokens, pos, builder, allowed_fields, depth)? {
            children.push(child);
        }
        match tokens.get(*pos) {
            Some(Token::Word(w)) if w == "AND" => {
                *pos += 1;
                if !tokens.get(*pos).is_some_and(is_operand_start) {
                    return Err(Fail); // dangling AND
                }
            }
            _ => break,
        }
    }
    Ok(match children.len() {
        0 => None,
        1 => Some(children[0]),
        _ => Some(builder.and(children)),
    })
}

/// unary := "NOT" unary | primary
fn parse_unary(
    tokens: &[Token],
    pos: &mut usize,
    builder: &mut QueryBuilder,
    allowed_fields: &[&str],
    depth: usize,
) -> ParseResult {
    if depth > MAX_PARSE_DEPTH {
        return Err(Fail); // too deep: demote to sanitized fallback
    }
    if let Some(Token::Word(w)) = tokens.get(*pos)
        && w == "NOT"
    {
        *pos += 1;
        let Some(child) = parse_unary(tokens, pos, builder, allowed_fields, depth + 1)? else {
            return Err(Fail); // dangling NOT (or NOT of empty group)
        };
        return Ok(Some(builder.not(child)));
    }
    parse_primary(tokens, pos, builder, allowed_fields, depth)
}

/// primary := "(" expr ")" | phrase | field:value | term
///
/// Returns `Ok(None)` for tokens with no searchable content (e.g. pure
/// punctuation), which callers silently skip.
fn parse_primary(
    tokens: &[Token],
    pos: &mut usize,
    builder: &mut QueryBuilder,
    allowed_fields: &[&str],
    depth: usize,
) -> ParseResult {
    match tokens.get(*pos) {
        Some(Token::LParen) => {
            *pos += 1;
            let inner = parse_or(tokens, pos, builder, allowed_fields, depth + 1)?;
            match tokens.get(*pos) {
                Some(Token::RParen) => {
                    *pos += 1;
                    Ok(inner)
                }
                _ => Err(Fail), // unbalanced parens
            }
        }
        Some(Token::Phrase(terms)) => {
            *pos += 1;
            let terms: Vec<Arc<str>> = terms.iter().map(|t| Arc::from(t.as_str())).collect();
            Ok(Some(builder.phrase(terms)))
        }
        Some(Token::Word(w)) if is_operand_start(&Token::Word(w.clone())) => {
            let w = w.clone();
            *pos += 1;
            Ok(word_to_node(&w, builder, allowed_fields))
        }
        _ => Err(Fail),
    }
}

/// Lower a single word token: allowlisted `field:value` becomes a fielded
/// term; anything else (unknown qualifiers, multi-colon tokens, stray
/// punctuation) demotes to plain terms.
fn word_to_node(
    word: &str,
    builder: &mut QueryBuilder,
    allowed_fields: &[&str],
) -> Option<leit_core::QueryNodeId> {
    if let Some((field, value)) = word.split_once(':')
        && allowed_fields.contains(&field)
        && !value.is_empty()
        && !value.contains(':')
    {
        return Some(builder.term_with_field(value, field));
    }
    // Demote: colons act as separators, each non-empty piece is a term.
    let pieces: Vec<&str> = word.split(':').filter(|p| !p.is_empty()).collect();
    match pieces.len() {
        0 => None,
        1 => Some(builder.term(pieces[0])),
        _ => {
            let children: Vec<_> = pieces.iter().map(|p| builder.term(*p)).collect();
            Some(builder.or(children))
        }
    }
}

/// Reduce arbitrary prose to an OR of parse-safe terms: keep
/// alphanumeric-cored tokens (allowing inner `_ . -`), drop operator
/// punctuation entirely. This is the guaranteed-total last resort.
fn fallback_terms(query: &str) -> Option<UserQueryProgram> {
    let mut builder = QueryBuilder::new();
    let terms: Vec<_> = query
        .split(|c: char| !(c.is_alphanumeric() || c == '_' || c == '.' || c == '-'))
        .map(|t| t.trim_matches(|c: char| !c.is_alphanumeric()))
        .filter(|t| !t.is_empty())
        .map(|t| builder.term(t))
        .collect();
    match terms.len() {
        0 => None,
        1 => builder.build(),
        _ => {
            let root = builder.or(terms);
            builder.set_root(root);
            builder.build()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use leit_index::{BooleanOp, UserQueryNode};

    const FIELDS: &[&str] = &["content"];

    fn root_node(program: &UserQueryProgram) -> &UserQueryNode {
        program.get(program.root()).expect("root exists")
    }

    #[test]
    fn plain_prose_becomes_or_of_terms() {
        let program = parse("jj safe-push worktree", FIELDS).expect("parses");
        match root_node(&program) {
            UserQueryNode::Boolean { op, children } => {
                assert_eq!(*op, BooleanOp::Or);
                assert_eq!(children.len(), 3);
            }
            other => panic!("expected OR root, got {other:?}"),
        }
    }

    #[test]
    fn single_word_becomes_term() {
        let program = parse("rust", FIELDS).expect("parses");
        match root_node(&program) {
            UserQueryNode::Term { term, field } => {
                assert_eq!(term.as_ref(), "rust");
                assert!(field.is_none());
            }
            other => panic!("expected term root, got {other:?}"),
        }
    }

    #[test]
    fn quoted_text_becomes_phrase() {
        let program = parse("\"exact words here\"", FIELDS).expect("parses");
        match root_node(&program) {
            UserQueryNode::Phrase { terms, slop } => {
                assert_eq!(terms.len(), 3);
                assert_eq!(*slop, 0);
            }
            other => panic!("expected phrase root, got {other:?}"),
        }
    }

    #[test]
    fn uppercase_operators_build_boolean_nodes() {
        let program = parse("rust AND memory", FIELDS).expect("parses");
        match root_node(&program) {
            UserQueryNode::Boolean { op, children } => {
                assert_eq!(*op, BooleanOp::And);
                assert_eq!(children.len(), 2);
            }
            other => panic!("expected AND root, got {other:?}"),
        }

        let program = parse("NOT rust", FIELDS).expect("parses");
        assert!(matches!(
            root_node(&program),
            UserQueryNode::Boolean {
                op: BooleanOp::Not,
                ..
            }
        ));
    }

    #[test]
    fn lowercase_operator_words_are_plain_terms() {
        let program = parse("cats or dogs", FIELDS).expect("parses");
        match root_node(&program) {
            UserQueryNode::Boolean { op, children } => {
                assert_eq!(*op, BooleanOp::Or);
                assert_eq!(children.len(), 3, "'or' is a term, not an operator");
            }
            other => panic!("expected OR root, got {other:?}"),
        }
    }

    #[test]
    fn balanced_parens_group() {
        let program = parse("(alpha beta) AND gamma", FIELDS).expect("parses");
        match root_node(&program) {
            UserQueryNode::Boolean { op, children } => {
                assert_eq!(*op, BooleanOp::And);
                assert_eq!(children.len(), 2);
                assert!(matches!(
                    program.get(children[0]),
                    Some(UserQueryNode::Boolean {
                        op: BooleanOp::Or,
                        ..
                    })
                ));
            }
            other => panic!("expected AND root, got {other:?}"),
        }
    }

    #[test]
    fn allowlisted_field_qualifier_becomes_fielded_term() {
        let program = parse("content:rust", FIELDS).expect("parses");
        match root_node(&program) {
            UserQueryNode::Term { term, field } => {
                assert_eq!(term.as_ref(), "rust");
                assert_eq!(field.as_deref(), Some("content"));
            }
            other => panic!("expected fielded term, got {other:?}"),
        }
    }

    #[test]
    fn unknown_qualifier_demotes_to_plain_terms() {
        let program = parse("authority:workers", FIELDS).expect("parses");
        match root_node(&program) {
            UserQueryNode::Boolean { op, children } => {
                assert_eq!(*op, BooleanOp::Or);
                assert_eq!(children.len(), 2, "both colon pieces become terms");
            }
            other => panic!("expected OR of demoted terms, got {other:?}"),
        }
    }

    #[test]
    fn prose_with_stray_colon_never_fails() {
        let program = parse("push authority: workers prepare commits", FIELDS).expect("parses");
        // "authority:" demotes to the term "authority"; nothing errors.
        match root_node(&program) {
            UserQueryNode::Boolean { op, children } => {
                assert_eq!(*op, BooleanOp::Or);
                assert_eq!(children.len(), 5);
            }
            other => panic!("expected OR root, got {other:?}"),
        }
    }

    #[test]
    fn adversarial_inputs_never_panic_or_fail_structurally() {
        // Structural garbage must either parse (demoted) or return None —
        // never panic. This is the totality contract.
        let cases = [
            "a:b:c",
            "((",
            "))((",
            "OR",
            "AND AND",
            "NOT",
            "NOT NOT",
            "--flags --another",
            "\"unterminated quote",
            "()",
            "( )",
            ":::",
            "::(){}[]",
            "field: :value",
            "日本語 テスト",
            "emoji 🦀 rust",
            "a OR (b AND",
            "OR OR OR",
            "NOT )",
            "\"\"",
            "((()))",
            "-",
            ".",
        ];
        for case in cases {
            let _ = parse(case, FIELDS); // must not panic
        }
    }

    #[test]
    fn pseudo_random_punctuation_soup_never_panics() {
        // Deterministic pseudo-random strings over an operator-heavy
        // alphabet: property-style totality check without a proptest dep.
        let alphabet: Vec<char> =
            "ab OR()\"':-NOTAND \u{2603}\u{1F980}".chars().collect();
        let mut state = 0x2545_F491_4F6C_DD1D_u64;
        for len in 0..64 {
            let mut s = String::new();
            for _ in 0..len {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                let idx = (state % alphabet.len() as u64) as usize;
                s.push(alphabet[idx]);
            }
            let _ = parse(&s, FIELDS); // must not panic
        }
    }

    #[test]
    fn empty_and_pure_punctuation_return_none() {
        assert!(parse("", FIELDS).is_none());
        assert!(parse("   ", FIELDS).is_none());
        assert!(parse(":::", FIELDS).is_none());
        assert!(parse("()", FIELDS).is_none());
    }

    #[test]
    fn unbalanced_parens_fall_back_to_terms() {
        let program = parse("(alpha beta", FIELDS).expect("falls back to terms");
        match root_node(&program) {
            UserQueryNode::Boolean { op, children } => {
                assert_eq!(*op, BooleanOp::Or);
                assert_eq!(children.len(), 2);
            }
            other => panic!("expected OR fallback, got {other:?}"),
        }
    }

    #[test]
    fn deep_not_chain_is_total() {
        // 100k-deep NOT chain must not overflow the stack; the depth bound
        // demotes it to the sanitized fallback (still total, still Some).
        let mut q = String::new();
        for _ in 0..100_000 {
            q.push_str("NOT ");
        }
        q.push('x');
        assert!(parse(&q, FIELDS).is_some());
    }

    #[test]
    fn deep_paren_nest_is_total() {
        // 100k-deep paren nesting must not overflow the stack.
        let mut q = "(".repeat(100_000);
        q.push('x');
        q.push_str(&")".repeat(100_000));
        assert!(parse(&q, FIELDS).is_some());
    }
}
