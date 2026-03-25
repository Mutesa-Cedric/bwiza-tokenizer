use bwiza_tokenizer_runtime::CRATE_NAME;

#[test]
fn crate_name_is_stable() {
    assert_eq!(CRATE_NAME, "bwiza_tokenizer_runtime");
}
