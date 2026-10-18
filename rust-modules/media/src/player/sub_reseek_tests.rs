//! FORK (RichiL10): the Off -> image-track re-read (`sub_reseek`, `player::reseek_for_image_subtitle`).
//!
//! An image subtitle picked from Off in direct play must not wait for the demux read-ahead (the
//! fork's 96 MiB video queue) to play out: the pick re-reads once from the playhead. Every other
//! pick (text, image -> image, the cold start, a remux, a short read-ahead, the knob off) leaves
//! the demuxer alone. Integer arithmetic only (the libm gate in `ci/check-deps.sh`).

use super::*;
use std::sync::atomic::Ordering::Relaxed;

const S: i64 = 1_000_000_000;

/// The player globals a re-read decision reads, set to "a direct play presenting at `pos_ns` with
/// the demuxer `ahead_ns` beyond it, image tracks `mask`", and put back on drop. The caller holds
/// `plx_base::testlock::serial()`.
pub(crate) struct Arranged {
    seen_frame: bool,
    seeking: bool,
    seek_display_ns: i64,
    playpos_ns: i64,
    video_tail_ns: i64,
    disp_base: i64,
    side_owner: bool,
    mask: u64,
    seek_to_ns: i64,
    seek_reqs: u32,
    sub_idx: i32,
    ended: bool,
}

impl Arranged {
    pub(crate) fn playing(pos_ns: i64, ahead_ns: i64, mask: u64) -> Self {
        let saved = Arranged {
            seen_frame: SHARED.seen_frame.load(Relaxed),
            seeking: SHARED.seeking.load(Relaxed),
            seek_display_ns: SHARED.seek_display_ns.load(Relaxed),
            playpos_ns: SHARED.playpos_ns.load(Relaxed),
            video_tail_ns: SHARED.hls_video_tail_ns.load(Relaxed),
            disp_base: SHARED.disp_base.load(Relaxed),
            side_owner: SHARED.side_subs_owner.load(Relaxed),
            mask: IMAGE_SUB_TRACKS.load(Relaxed),
            seek_to_ns: TX.seek_to_ns.load(Relaxed),
            seek_reqs: TX.seek_reqs.load(Relaxed),
            sub_idx: SHARED.desired_sub_idx.load(Relaxed),
            ended: SHARED.ended.load(Relaxed),
        };
        SHARED.seen_frame.store(true, Relaxed);
        SHARED.seeking.store(false, Relaxed);
        SHARED.seek_display_ns.store(-1, Relaxed);
        SHARED.playpos_ns.store(pos_ns, Relaxed);
        SHARED.hls_video_tail_ns.store(if ahead_ns < 0 { -1 } else { pos_ns + ahead_ns }, Relaxed);
        SHARED.disp_base.store(0, Relaxed);
        SHARED.side_subs_owner.store(false, Relaxed);
        publish_image_sub_tracks(mask);
        TX.seek_to_ns.store(-1, Relaxed);
        TX.seek_reqs.store(0, Relaxed);
        SHARED.desired_sub_idx.store(-1, Relaxed);
        saved
    }

    /// Seek requests made since the arrangement.
    pub(crate) fn seeks(&self) -> u32 {
        TX.seek_reqs.load(Relaxed)
    }
}

impl Drop for Arranged {
    fn drop(&mut self) {
        crate::route::reject_user_seek();
        SHARED.seen_frame.store(self.seen_frame, Relaxed);
        SHARED.seeking.store(self.seeking, Relaxed);
        SHARED.seek_display_ns.store(self.seek_display_ns, Relaxed);
        SHARED.playpos_ns.store(self.playpos_ns, Relaxed);
        SHARED.hls_video_tail_ns.store(self.video_tail_ns, Relaxed);
        SHARED.disp_base.store(self.disp_base, Relaxed);
        SHARED.side_subs_owner.store(self.side_owner, Relaxed);
        IMAGE_SUB_TRACKS.store(self.mask, Relaxed);
        TX.seek_to_ns.store(self.seek_to_ns, Relaxed);
        TX.seek_reqs.store(self.seek_reqs, Relaxed);
        SHARED.desired_sub_idx.store(self.sub_idx, Relaxed);
        SHARED.ended.store(self.ended, Relaxed);
    }
}

/// Track 0 text, tracks 1 and 2 image (the shape of a Blu-ray rip with an SRT added).
const MASK: u64 = 0b110;

fn no_knob_file<R>(f: impl FnOnce() -> R) -> R {
    plx_base::forkknobs::with_conf_for_test(None, f)
}

#[test]
fn off_to_an_image_track_rereads_from_the_playhead_exactly_once() {
    let _g = plx_base::testlock::serial();
    let a = Arranged::playing(736 * S, 12 * S, MASK);
    no_knob_file(|| {
        assert!(reseek_for_image_subtitle(-1, 1, true), "Off -> image with 12 s read ahead");
        assert_eq!(a.seeks(), 1);
        assert_eq!(TX.seek_to_ns.load(Relaxed), 736 * S, "the target is the playhead itself");
        // The same pick again while that seek resolves (a double OK) asks for nothing more.
        assert!(!reseek_for_image_subtitle(-1, 1, true));
        assert_eq!(a.seeks(), 1, "exactly one re-read");
    });
}

#[test]
fn off_to_a_text_track_never_rereads() {
    let _g = plx_base::testlock::serial();
    let a = Arranged::playing(736 * S, 12 * S, MASK);
    no_knob_file(|| {
        assert!(!reseek_for_image_subtitle(-1, 0, true), "text is decoded while Off already");
        assert_eq!(a.seeks(), 0);
    });
}

#[test]
fn image_to_image_while_on_never_rereads() {
    let _g = plx_base::testlock::serial();
    let a = Arranged::playing(736 * S, 12 * S, MASK);
    no_knob_file(|| {
        assert!(!reseek_for_image_subtitle(1, 2, true), "every image track is decoded while on");
        assert!(!reseek_for_image_subtitle(0, 1, true), "text -> image: images were decoded already");
        assert_eq!(a.seeks(), 0);
    });
}

#[test]
fn the_knob_off_never_rereads() {
    let _g = plx_base::testlock::serial();
    let a = Arranged::playing(736 * S, 12 * S, MASK);
    plx_base::forkknobs::with_conf_for_test(Some("sub_reseek=off\n"), || {
        assert!(!reseek_for_image_subtitle(-1, 1, true));
    });
    assert_eq!(a.seeks(), 0, "sub_reseek=off restores upstream's wait");
    plx_base::forkknobs::with_conf_for_test(Some("sub_reseek=on\n"), || {
        assert!(reseek_for_image_subtitle(-1, 1, true), "the explicit default re-reads");
    });
    assert_eq!(a.seeks(), 1);
}

#[test]
fn a_remux_transcode_or_side_reader_route_never_rereads() {
    let _g = plx_base::testlock::serial();
    let a = Arranged::playing(736 * S, 12 * S, MASK);
    no_knob_file(|| {
        assert!(!reseek_for_image_subtitle(-1, 1, false), "the server's stream is not ours to re-read");
        SHARED.side_subs_owner.store(true, Relaxed);
        assert!(!reseek_for_image_subtitle(-1, 1, true), "the side reader draws these subtitles");
        assert_eq!(a.seeks(), 0);
    });
}

#[test]
fn a_small_or_unknown_read_ahead_never_rereads() {
    let _g = plx_base::testlock::serial();
    let a = Arranged::playing(736 * S, 2 * S, MASK);
    no_knob_file(|| {
        assert!(!reseek_for_image_subtitle(-1, 1, true), "2 s ahead: upstream's case, already fine");
        SHARED.hls_video_tail_ns.store(736 * S + SUB_RESEEK_MIN_READ_AHEAD_MS * 1_000_000, Relaxed);
        assert!(!reseek_for_image_subtitle(-1, 1, true), "exactly the threshold");
        SHARED.hls_video_tail_ns.store(-1, Relaxed);
        assert!(!reseek_for_image_subtitle(-1, 1, true), "no packet since the open or seek");
        assert_eq!(a.seeks(), 0);
    });
}

#[test]
fn the_cold_start_and_a_pending_seek_never_reread() {
    let _g = plx_base::testlock::serial();
    let a = Arranged::playing(736 * S, 12 * S, MASK);
    no_knob_file(|| {
        SHARED.seen_frame.store(false, Relaxed);
        assert!(!reseek_for_image_subtitle(-1, 1, true), "no picture yet: the demux starts with it on");
        SHARED.seen_frame.store(true, Relaxed);
        TX.seek_to_ns.store(900 * S, Relaxed);
        assert!(!reseek_for_image_subtitle(-1, 1, true), "a seek already restarts the demux");
        assert_eq!(a.seeks(), 0);
    });
}

#[test]
fn the_decision_is_the_conjunction_of_its_facts() {
    let due = SubReseekFacts {
        knob_on: true,
        direct: true,
        side_reader: false,
        prev_idx: -1,
        new_idx: 2,
        image_mask: MASK,
        read_ahead_ms: Some(SUB_RESEEK_MIN_READ_AHEAD_MS + 1),
        seen_frame: true,
        seeking: false,
    };
    assert!(sub_reseek_due(&due));
    let each_alone_refuses = [
        SubReseekFacts { knob_on: false, ..due },
        SubReseekFacts { direct: false, ..due },
        SubReseekFacts { side_reader: true, ..due },
        SubReseekFacts { prev_idx: 1, ..due },
        SubReseekFacts { new_idx: -1, ..due },
        SubReseekFacts { new_idx: 0, ..due },
        SubReseekFacts { new_idx: 64, ..due },
        SubReseekFacts { read_ahead_ms: None, ..due },
        SubReseekFacts { read_ahead_ms: Some(SUB_RESEEK_MIN_READ_AHEAD_MS), ..due },
        SubReseekFacts { seen_frame: false, ..due },
        SubReseekFacts { seeking: true, ..due },
    ];
    for (i, f) in each_alone_refuses.iter().enumerate() {
        assert!(!sub_reseek_due(f), "case {i}: {f:?}");
    }
}

#[test]
fn the_image_mask_marks_image_tracks_in_file_order() {
    assert_eq!(image_sub_mask([false, true, true]), 0b110);
    assert_eq!(image_sub_mask(std::iter::empty::<bool>()), 0);
    let many = image_sub_mask((0..70).map(|i| i == 63 || i == 65));
    assert_eq!(many, 1u64 << 63, "a track past bit 63 is never marked");
    assert!(is_image_sub_track(63, many));
    assert!(!is_image_sub_track(64, u64::MAX));
    assert!(!is_image_sub_track(-1, u64::MAX));
}
