//! SQLite provider for [`CapsuleStore`], implemented on [`Database`].
//!
//! Bodies are stored as the exact canonical JSON bytes that were
//! fingerprinted, so reads verify integrity by recomputing the fingerprint
//! before deserializing. Rows are append-only; `(capsule_id, fingerprint)`
//! is unique and re-puts are no-ops.

use crate::capsule::{CapsuleKind, CapsuleRef, CapsuleStore, ProjectCapsuleV1, capsule_uri};
use crate::db::Database;
use crate::error::GraphRagError;
use rusqlite::{OptionalExtension, params};
use sha2::{Digest, Sha256};

fn fingerprint_bytes(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    let mut hex = String::with_capacity(71);
    hex.push_str("sha256:");
    for b in digest {
        use std::fmt::Write;
        let _ = write!(hex, "{b:02x}");
    }
    hex
}

fn kind_from_str(kind: &str) -> Result<CapsuleKind, GraphRagError> {
    match kind {
        "project" => Ok(CapsuleKind::Project),
        "segment" => Ok(CapsuleKind::Segment),
        "portfolio" => Ok(CapsuleKind::Portfolio),
        other => Err(GraphRagError::Capsule(format!(
            "stored capsule row has unknown kind {other:?}"
        ))),
    }
}

impl Database {
    fn capsule_row_to_ref(
        kind: &str,
        capsule_id: String,
        project_id: String,
        content_fingerprint: String,
        generated_at: String,
    ) -> Result<CapsuleRef, GraphRagError> {
        let kind = kind_from_str(kind)?;
        Ok(CapsuleRef {
            schema_version: 1,
            uri: capsule_uri(kind, &capsule_id),
            kind,
            capsule_id,
            project_id,
            content_fingerprint,
            generated_at,
        })
    }

    fn load_capsule_body(
        &self,
        body: &str,
        expected_fp: &str,
    ) -> Result<ProjectCapsuleV1, GraphRagError> {
        let actual = fingerprint_bytes(body.as_bytes());
        if actual != expected_fp {
            return Err(GraphRagError::Capsule(format!(
                "stored capsule bytes do not match fingerprint (expected {expected_fp}, got {actual})"
            )));
        }
        let capsule: ProjectCapsuleV1 = serde_json::from_str(body)?;
        Ok(capsule)
    }
}

impl CapsuleStore for Database {
    type Error = GraphRagError;

    fn put_capsule(&self, capsule: &ProjectCapsuleV1) -> Result<CapsuleRef, GraphRagError> {
        let reference = capsule
            .reference()
            .map_err(|e| GraphRagError::Capsule(e.to_string()))?;
        let body = serde_json::to_string(capsule)?;
        debug_assert_eq!(
            fingerprint_bytes(body.as_bytes()),
            reference.content_fingerprint
        );
        self.conn().execute(
            "INSERT INTO capsules (capsule_id, project_id, kind, content_fingerprint, generated_at, body)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6)
             ON CONFLICT(capsule_id, content_fingerprint) DO NOTHING",
            params![
                reference.capsule_id,
                reference.project_id,
                reference.kind.as_str(),
                reference.content_fingerprint,
                reference.generated_at,
                body,
            ],
        )?;
        Ok(reference)
    }

    fn latest_capsule(&self, capsule_id: &str) -> Result<Option<ProjectCapsuleV1>, GraphRagError> {
        let row: Option<(String, String)> = self
            .conn()
            .query_row(
                "SELECT body, content_fingerprint FROM capsules
                 WHERE capsule_id = ?1 ORDER BY id DESC LIMIT 1",
                params![capsule_id],
                |r| Ok((r.get(0)?, r.get(1)?)),
            )
            .optional()?;
        row.map(|(body, fp)| self.load_capsule_body(&body, &fp))
            .transpose()
    }

    fn capsule_by_fingerprint(
        &self,
        capsule_id: &str,
        content_fingerprint: &str,
    ) -> Result<Option<ProjectCapsuleV1>, GraphRagError> {
        let row: Option<(String, String)> = self
            .conn()
            .query_row(
                "SELECT body, content_fingerprint FROM capsules
                 WHERE capsule_id = ?1 AND content_fingerprint = ?2",
                params![capsule_id, content_fingerprint],
                |r| Ok((r.get(0)?, r.get(1)?)),
            )
            .optional()?;
        row.map(|(body, fp)| self.load_capsule_body(&body, &fp))
            .transpose()
    }

    fn capsule_history(&self, capsule_id: &str) -> Result<Vec<CapsuleRef>, GraphRagError> {
        let mut stmt = self.conn().prepare(
            "SELECT kind, capsule_id, project_id, content_fingerprint, generated_at
             FROM capsules WHERE capsule_id = ?1 ORDER BY id DESC",
        )?;
        let rows = stmt.query_map(params![capsule_id], |r| {
            Ok((
                r.get::<_, String>(0)?,
                r.get::<_, String>(1)?,
                r.get::<_, String>(2)?,
                r.get::<_, String>(3)?,
                r.get::<_, String>(4)?,
            ))
        })?;
        let mut out = Vec::new();
        for row in rows {
            let (kind, cid, pid, fp, gen_at) = row?;
            out.push(Self::capsule_row_to_ref(&kind, cid, pid, fp, gen_at)?);
        }
        Ok(out)
    }

    fn list_capsules(&self, project_id: Option<&str>) -> Result<Vec<CapsuleRef>, GraphRagError> {
        let sql = "SELECT kind, capsule_id, project_id, content_fingerprint, generated_at
             FROM capsules WHERE id IN (SELECT MAX(id) FROM capsules GROUP BY capsule_id)
             AND (?1 IS NULL OR project_id = ?1)
             ORDER BY capsule_id";
        let mut stmt = self.conn().prepare(sql)?;
        let rows = stmt.query_map(params![project_id], |r| {
            Ok((
                r.get::<_, String>(0)?,
                r.get::<_, String>(1)?,
                r.get::<_, String>(2)?,
                r.get::<_, String>(3)?,
                r.get::<_, String>(4)?,
            ))
        })?;
        let mut out = Vec::new();
        for row in rows {
            let (kind, cid, pid, fp, gen_at) = row?;
            out.push(Self::capsule_row_to_ref(&kind, cid, pid, fp, gen_at)?);
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn corrupted_body_is_detected_on_read() {
        let dir = tempfile::TempDir::new().unwrap();
        let db = Database::open(&dir.path().join("t.db")).unwrap();
        db.conn()
            .execute(
                "INSERT INTO capsules (capsule_id, project_id, kind, content_fingerprint, generated_at, body)
                 VALUES ('x', 'p', 'project', 'sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa', '2026-08-07T00:00:00Z', '{}')",
                [],
            )
            .unwrap();
        let err = db.latest_capsule("x").unwrap_err();
        assert!(
            err.to_string().contains("do not match fingerprint"),
            "{err}"
        );
    }
}
