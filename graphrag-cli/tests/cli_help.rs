use assert_cmd::Command;
use predicates::str::contains;

#[test]
fn help_lists_core_subcommands() {
    let mut cmd = Command::cargo_bin("graphrag").expect("binary exists");
    cmd.arg("--help");
    cmd.assert()
        .success()
        .stdout(contains("note"))
        .stdout(contains("ask"))
        .stdout(contains("log"))
        .stdout(contains("enrich"));
}
