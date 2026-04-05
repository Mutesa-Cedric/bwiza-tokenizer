pub const BYTE_FALLBACK_PREFIX: char = '\u{E000}';
pub const BYTE_FALLBACK_COUNT: usize = 256;

pub fn byte_fallback_piece(byte: u8) -> String {
    format!("{BYTE_FALLBACK_PREFIX}{byte:02X}")
}

pub fn parse_byte_fallback_piece(piece: &str) -> Option<u8> {
    if piece.chars().count() != 3 || !piece.starts_with(BYTE_FALLBACK_PREFIX) {
        return None;
    }

    u8::from_str_radix(&piece[BYTE_FALLBACK_PREFIX.len_utf8()..], 16).ok()
}

pub fn is_byte_fallback_piece(piece: &str) -> bool {
    parse_byte_fallback_piece(piece).is_some()
}

pub fn decode_byte_fallback_pieces(pieces: &[&str]) -> String {
    let mut output = String::new();
    let mut pending_bytes = Vec::new();

    let flush_pending_bytes = |output: &mut String, pending_bytes: &mut Vec<u8>| {
        if pending_bytes.is_empty() {
            return;
        }

        output.push_str(String::from_utf8_lossy(pending_bytes).as_ref());
        pending_bytes.clear();
    };

    for piece in pieces {
        if let Some(byte) = parse_byte_fallback_piece(piece) {
            pending_bytes.push(byte);
            continue;
        }

        flush_pending_bytes(&mut output, &mut pending_bytes);
        output.push_str(piece);
    }

    flush_pending_bytes(&mut output, &mut pending_bytes);
    output
}
