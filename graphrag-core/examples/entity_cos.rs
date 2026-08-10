fn main() {
    let terms: Vec<String> = std::env::args().skip(1).collect();
    let e = graphrag_core::Embedder::new(Default::default()).expect("embedder");
    let vecs: Vec<Vec<f32>> = terms.iter().map(|t| e.embed(t).expect("embed")).collect();
    let cos = |a: &[f32], b: &[f32]| {
        let (mut d, mut na, mut nb) = (0f32, 0f32, 0f32);
        for (x, y) in a.iter().zip(b) {
            d += x * y;
            na += x * x;
            nb += y * y;
        }
        d / (na.sqrt() * nb.sqrt())
    };
    print!("{:26}", "");
    for t in &terms {
        print!("{:>22}", &t[..t.len().min(20)]);
    }
    println!();
    for (i, a) in vecs.iter().enumerate() {
        print!("{:26}", &terms[i][..terms[i].len().min(24)]);
        for b in &vecs {
            print!("{:>22.3}", cos(a, b));
        }
        println!();
    }
}
