//! FORK (RichiL10): runtime knobs for the household television.
//!
//! The release build has no `/tmp` developer triggers (see [`crate::devtrig`]), so the fork's
//! own playback choices that need an A/B on the real set without a rebuild are read from ONE
//! plain file instead: `/media/developer/plxnative-fork.conf`, persistent across reboots,
//! `key=value` per line, `#` starts a comment. A missing or unreadable file means "every knob at
//! its default", and every default is the behaviour the fork ships, so deleting the file is
//! always the way back.
//!
//! Nothing here is security-relevant on purpose: a knob selects between implemented playback
//! behaviours (an audio sample format, whether a remux declares Dolby Vision, ...). It never
//! carries a token, a URL, a path or a server, which is exactly the class `devtrig` keeps out of
//! public builds.
//!
//! Known keys (the consumers own the meaning; unknown keys are ignored):
//!
//! | key            | values (default first)        | consumer |
//! |----------------|-------------------------------|----------|
//! | `remux_dv`     | `on`, `off`                   | route: keep the source's Dolby Vision on a server remux with video copy |
//! | `encode_audio` | `auto`, `ac3`, `eac3`         | profile: the codec PMS encodes a non-direct-playable track to (`auto` = by TV sound output) |
//! | `pcm_hi`       | `s32`, `s24`, `s16`           | feed: PCM sample format for sources deeper than 16 bit |
//! | `flac`         | `pcm`, `off`                  | feed: FLAC decoded in-app to PCM, or left to the server |
//! | `opus`         | `native`, `off`               | feed: Opus fed as-is, or left to the server |
//! | `dts_core`     | `auto`, `never`               | gate: DTS-HD above the TV's DTS channel ceiling plays as its core (`never` = the server converts it) |
//!
//! The file is re-read when its modification time changes, at most once per second, so a change
//! takes effect on the next playback without restarting the app.

use std::collections::BTreeMap;
use std::sync::Mutex;
use std::time::{Duration, Instant, SystemTime};

/// The persistent location: `/media/developer` survives reboots and is writable from the root
/// shell the fork is maintained through. Host tests point it elsewhere with `PLXNATIVE_FORK_CONF`.
const DEFAULT_PATH: &str = "/media/developer/plxnative-fork.conf";
const RECHECK: Duration = Duration::from_secs(1);

struct Cache {
    checked: Option<Instant>,
    mtime: Option<SystemTime>,
    path: String,
    map: BTreeMap<String, String>,
}

static CACHE: Mutex<Cache> = Mutex::new(Cache {
    checked: None,
    mtime: None,
    path: String::new(),
    map: BTreeMap::new(),
});

fn conf_path() -> String {
    std::env::var("PLXNATIVE_FORK_CONF").unwrap_or_else(|_| DEFAULT_PATH.to_string())
}

/// `key=value` lines; keys and values trimmed and lowercased; `#` comments; blank and malformed
/// lines skipped. Pure, so the format is host-testable.
pub fn parse(text: &str) -> BTreeMap<String, String> {
    let mut map = BTreeMap::new();
    for line in text.lines() {
        let line = line.split('#').next().unwrap_or("").trim();
        if let Some((k, v)) = line.split_once('=') {
            let (k, v) = (k.trim().to_ascii_lowercase(), v.trim().to_ascii_lowercase());
            if !k.is_empty() && !v.is_empty() {
                map.insert(k, v);
            }
        }
    }
    map
}

/// The knob's value, or `None` when the file does not set it (the consumer's default applies).
pub fn get(key: &str) -> Option<String> {
    let mut c = CACHE.lock().unwrap_or_else(|e| e.into_inner());
    let path = conf_path();
    let stale = c.path != path || c.checked.is_none_or(|t| t.elapsed() >= RECHECK);
    if stale {
        c.checked = Some(Instant::now());
        let mtime = std::fs::metadata(&path).and_then(|m| m.modified()).ok();
        if c.path != path || mtime != c.mtime {
            c.map = match mtime {
                Some(_) => std::fs::read_to_string(&path).map(|s| parse(&s)).unwrap_or_default(),
                None => BTreeMap::new(),
            };
            c.mtime = mtime;
            c.path = path;
        }
    }
    c.map.get(key).cloned()
}

/// `get(key) == value`, with a missing knob answering `value == default`.
pub fn is(key: &str, value: &str, default: &str) -> bool {
    match get(key) {
        Some(v) => v == value,
        None => value == default,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_reads_pairs_comments_and_case() {
        let m = parse("# fork knobs\nPCM_HI = S16 \n\nflac=off # server\nbroken line\n=x\ny=\n");
        assert_eq!(m.get("pcm_hi").map(String::as_str), Some("s16"));
        assert_eq!(m.get("flac").map(String::as_str), Some("off"));
        assert_eq!(m.len(), 2);
    }

    #[test]
    fn missing_file_means_defaults_and_a_change_is_picked_up() {
        let _g = crate::testlock::serial();
        let dir = std::env::temp_dir().join(format!("plx-forkknobs-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let file = dir.join("fork.conf");
        let _ = std::fs::remove_file(&file);
        std::env::set_var("PLXNATIVE_FORK_CONF", &file);
        assert_eq!(get("flac"), None);
        assert!(is("flac", "pcm", "pcm"));
        std::fs::write(&file, "flac=off\n").unwrap();
        // Force the next read past the once-per-second recheck window.
        CACHE.lock().unwrap().checked = None;
        assert_eq!(get("flac").as_deref(), Some("off"));
        assert!(!is("flac", "pcm", "pcm"));
        std::env::remove_var("PLXNATIVE_FORK_CONF");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
