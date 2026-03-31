use unicode_normalization::UnicodeNormalization;

const BOUNDARY_MARKER: char = '▁';

pub fn normalize_text(input: &str) -> String {
    let nfc_text: String = input.nfc().collect();
    let mut collapsed = String::new();
    let mut last_was_space = false;

    for ch in nfc_text.chars() {
        if ch.is_whitespace() {
            if !last_was_space {
                collapsed.push(' ');
                last_was_space = true;
            }
        } else {
            collapsed.push(ch);
            last_was_space = false;
        }
    }

    let trimmed = collapsed.trim_matches(' ');
    if trimmed.is_empty() {
        return String::new();
    }

    let mut output = String::with_capacity(trimmed.len() + 1);
    output.push(BOUNDARY_MARKER);

    for ch in trimmed.chars() {
        if ch == ' ' {
            output.push(BOUNDARY_MARKER);
        } else {
            output.push(ch);
        }
    }

    output
}
