//! FORK (RichiL10): runtime knobs for the household television.
//!
//! The release build has no `/tmp` developer triggers (see [`crate::devtrig`]), so the fork's
//! own playback choices that need an A/B on the real set without a rebuild are read from ONE
//! plain file instead: `/media/developer/plxnative-fork.conf`, persistent across reboots,
//! `key=value` per line, `#` starts a comment. A missing or unreadable file means "every knob at
//! its default", and every default is the behaviour the fork ships, so deleting the file is
//! always the way back. The 0.8 fork install reads the same file with the same keys, so one file
//! serves both installs.
//!
//! Nothing here is security-relevant on purpose: a knob selects between implemented playback
//! behaviours (an audio sample format, whether a lane is used, ...). It never carries a token, a
//! URL, a path or a server, which is exactly the class `devtrig` keeps out of public builds.
//!
//! Known keys (the consumers own the meaning; unknown keys are ignored):
//!
//! | key            | values (default first)        | consumer |
//! |----------------|-------------------------------|----------|
//! | `pcm_hi`       | `s32`, `s24`, `s16`           | feed: PCM sample format for sources deeper than 16 bit |
//! | `flac`         | `pcm`, `off`                  | feed: FLAC decoded in-app to PCM, or left to the server |
//! | `opus`         | `native`, `off`               | feed: Opus fed as-is, or left to the server |
//! | `dts_core`     | `auto`, `never`               | gate: DTS-HD above the TV's DTS channel ceiling plays as its core (`never` = the server converts it) |
//! | `encode`       | `ac3`, `off`                  | feed: FLAC/PCM above the PCM lane's channels and TrueHD/MLP (48 kHz) decoded and encoded to AC3 5.1 640 kbps in-app (`off` = the server converts them) |
//! | `remux_dv`     | `on`, `off`                   | route: a server remux that copies the HEVC video declares the source's Dolby Vision as direct play does (`off` = declare nothing) |
//! | `encode_audio` | `auto`, `ac3`, `eac3`         | route: what the server encodes a track the TV cannot take into, off an optical output (`auto` = E-AC3 then AC3); optical is always AC3, and so is an output the TV did not report unless this says `eac3` |
//! | `sub_burn`     | `never4k`, `allow`            | route: never have the server burn a subtitle into a source wider than 1920 or with Dolby Vision/HDR (`allow` = upstream's burn) |
//! | `sub_reseek`   | `on`, `off`                   | player: an image subtitle (PGS/VobSub/DVB) picked from Off in direct play re-reads from the playhead so its first cue shows at once (`off` = wait for the read-ahead to play out) |
//!
//! The file is re-read when its modification time changes, at most once per second, so a change
//! takes effect on the next playback without restarting the app.

use std::collections::BTreeMap;
use std::sync::Mutex;
use std::time::{Duration, Instant, SystemTime};

/// The persistent location: `/media/developer` survives reboots and is writable from the root
/// shell the fork is maintained through. Tests (and the host simulator) point it elsewhere with
/// `PLXNATIVE_FORK_CONF`; a shipping build cannot be steered by the environment.
const DEFAULT_PATH: &str = "/media/developer/plxnative-fork.conf";
const RECHECK: Duration = Duration::from_secs(1);
const STEERABLE: bool = cfg!(any(test, feature = "test-support", feature = "hostsim"));

struct Cache {
    checked: Option<Instant>,
    mtime: Option<SystemTime>,
    path: String,
    map: BTreeMap<String, String>,
}

static CACHE: Mutex<Cache> = Mutex::new(Cache { checked: None, mtime: None, path: String::new(), map: BTreeMap::new() });

fn conf_path() -> String {
    if STEERABLE {
        if let Ok(p) = std::env::var("PLXNATIVE_FORK_CONF") {
            return p;
        }
    }
    DEFAULT_PATH.to_string()
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

/// Test fixture: run `f` with the knob file holding exactly `text` (`None` = no file), then put
/// the environment back. Callers hold `testlock::serial()`, the lock every process-global uses.
#[cfg(any(test, feature = "test-support"))]
pub fn with_conf_for_test<R>(text: Option<&str>, f: impl FnOnce() -> R) -> R {
    let dir = crate::testscratch::process_dir("plx-forkknobs");
    let file = dir.join("fork.conf");
    let _ = std::fs::remove_file(&file);
    if let Some(t) = text {
        std::fs::write(&file, t).expect("write fork.conf fixture");
    }
    let prev = std::env::var_os("PLXNATIVE_FORK_CONF");
    std::env::set_var("PLXNATIVE_FORK_CONF", &file);
    // A rewrite inside the once-per-second window must still be read.
    CACHE.lock().unwrap_or_else(|e| e.into_inner()).path.clear();
    let out = f();
    match prev {
        Some(p) => std::env::set_var("PLXNATIVE_FORK_CONF", p),
        None => std::env::remove_var("PLXNATIVE_FORK_CONF"),
    }
    let _ = std::fs::remove_file(&file);
    CACHE.lock().unwrap_or_else(|e| e.into_inner()).path.clear();
    out
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
        with_conf_for_test(None, || {
            assert_eq!(get("flac"), None);
            assert!(is("flac", "pcm", "pcm"));
        });
        with_conf_for_test(Some("flac=off\n"), || {
            assert_eq!(get("flac").as_deref(), Some("off"));
            assert!(!is("flac", "pcm", "pcm"));
        });
        with_conf_for_test(None, || assert!(is("encode", "ac3", "ac3")));
    }

    #[test]
    fn a_shipping_build_reads_only_the_fixed_path() {
        let _g = crate::testlock::serial();
        let prev = std::env::var_os("PLXNATIVE_FORK_CONF");
        std::env::remove_var("PLXNATIVE_FORK_CONF");
        assert_eq!(conf_path(), "/media/developer/plxnative-fork.conf");
        if let Some(p) = prev {
            std::env::set_var("PLXNATIVE_FORK_CONF", p);
        }
    }
}
