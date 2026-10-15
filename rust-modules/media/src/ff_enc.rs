//! FORK (RichiL10): the AC3 encode lane — audio the television cannot decode, converted in-app.
//!
//! The LG OLED B4 decodes FLAC and PCM in stereo only and has no TrueHD decoder at all, and its
//! optical output carries AC3 5.1 at most. Until this lane, a FLAC 5.1/7.1, a PCM track above
//! two channels or a TrueHD track was converted by the Plex server — a remux that drags subtitle
//! burn-in and every track switch through a server with no hardware transcoder. Here instead the
//! demux thread:
//!
//! 1. decodes the track with the bundled FFmpeg (`flac`, `truehd`, `mlp`; native `A_PCM/*` is
//!    read as-is through the PCM lane's [`SrcFmt`]);
//! 2. folds it to 5.1(side) — FL FR FC LFE SL SR — by channel ROLE, in Q15 fixed point with
//!    saturation ([`fold_matrix`], [`fold`]);
//! 3. cuts exact 1536-sample frames with a timestamped FIFO ([`Fifo`]) and encodes them with
//!    `ac3_fixed` at 640 kbps, the same target and quality PMS produces for this television, the
//!    Dolby Digital maximum and the most the optical link carries;
//! 4. hands each AC3 packet to the audio queue with its own pts, exactly as a native AC3 5.1
//!    48 kHz track is fed — the Load declares it as one (`player::engine::AudioLaneInfo`).
//!
//! Integer arithmetic only on the fold (the build is soft-float ARM, and `ac3_fixed` is the
//! integer encoder); no resampler, so the lane takes 48 kHz sources only and the route leaves
//! anything else with the server (`devcaps::Caps::audio_lane`).

use super::pcm::SrcFmt;
use super::*;

/// The encode lane's fixed output: AC3 5.1(side), 48 kHz, 640 kbps.
pub const ENC_RATE: u32 = 48_000;
pub const ENC_BITRATE: u32 = 640_000;
/// Samples per AC3 frame (six 256-sample blocks).
pub const AC3_FRAME: usize = 1536;
/// The encoder's priming (`AVCodecContext.initial_padding`): packet k carries the frame that
/// starts 256 samples before input frame k.
pub const AC3_PRIMING: i64 = 256;
const OUT_CH: usize = 6;
/// A packet whose pts is further than this from the FIFO's running clock re-anchors it.
const JUMP_NS: i64 = 100_000_000;

/// What the Load declared for the encode lane: the source's channel count, as PMS reported it
/// (the lane folds whatever the decoder actually returns). Published by the engine BEFORE the
/// demux thread exists, read once by it at open — the same contract as `pcm::declare`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EncSpec {
    pub channels: u32,
}

static DECLARED: std::sync::Mutex<Option<EncSpec>> = std::sync::Mutex::new(None);

pub fn declare(spec: Option<EncSpec>) {
    *DECLARED.lock().unwrap_or_else(|e| e.into_inner()) = spec;
}

pub(super) fn declared() -> Option<EncSpec> {
    *DECLARED.lock().unwrap_or_else(|e| e.into_inner())
}

/// `samples` at 48 kHz in ns. 1536 samples are exactly 32 ms, so frame pts never drift.
#[inline]
pub(super) fn samples_ns(samples: i64) -> i64 {
    (samples * 1_000_000_000).div_euclid(i64::from(ENC_RATE))
}

// ---- the fold -------------------------------------------------------------------------------

// enum AVChannel (libavutil/channel_layout.h), asserted in ci/ffabi-assert.c.
const CH_FL: i32 = 0;
const CH_FR: i32 = 1;
const CH_FC: i32 = 2;
const CH_LFE: i32 = 3;
const CH_BL: i32 = 4;
const CH_BR: i32 = 5;
const CH_FLC: i32 = 6;
const CH_FRC: i32 = 7;
const CH_BC: i32 = 8;
const CH_SL: i32 = 9;
const CH_SR: i32 = 10;
const CH_TC: i32 = 11;
const CH_TFL: i32 = 12;
const CH_TFC: i32 = 13;
const CH_TFR: i32 = 14;
const CH_TBL: i32 = 15;
const CH_TBC: i32 = 16;
const CH_TBR: i32 = 17;
const CH_STEREO_LEFT: i32 = 29;
const CH_STEREO_RIGHT: i32 = 30;
const CH_WIDE_LEFT: i32 = 31;
const CH_WIDE_RIGHT: i32 = 32;
const CH_SDL: i32 = 33;
const CH_SDR: i32 = 34;
const CH_LFE2: i32 = 35;
const CH_TSL: i32 = 36;
const CH_TSR: i32 = 37;
const CH_BFC: i32 = 38;
const CH_BFL: i32 = 39;
const CH_BFR: i32 = 40;
const CH_SSL: i32 = 41;
const CH_SSR: i32 = 42;
const CH_TSUL: i32 = 43;
const CH_TSUR: i32 = 44;

// Output slots, in 5.1(side) order — the order `ac3_fixed` takes its planes in.
const O_FL: usize = 0;
const O_FR: usize = 1;
const O_FC: usize = 2;
const O_LFE: usize = 3;
const O_SL: usize = 4;
const O_SR: usize = 5;

/// Gains in Q15: 1.0, and 1/sqrt(2) (0.70711 x 32768 = 23170.5).
pub(super) const Q15_ONE: i32 = 1 << 15;
pub(super) const Q15_M3DB: i32 = 23_170;

/// One input channel's contribution to each of the six outputs, Q15.
pub(super) type FoldRow = [i32; OUT_CH];

/// The channel roles of a layout with no stated order (Matroska `A_PCM/*` says only a count):
/// `av_channel_layout_default`'s choice for that count, so the lane reads such a track the way
/// FFmpeg itself would.
pub(super) fn default_roles(channels: usize) -> Vec<i32> {
    match channels {
        1 => vec![CH_FC],
        2 => vec![CH_FL, CH_FR],
        3 => vec![CH_FL, CH_FR, CH_LFE],
        4 => vec![CH_FL, CH_FR, CH_FC, CH_BC],
        5 => vec![CH_FL, CH_FR, CH_FC, CH_BL, CH_BR],
        6 => vec![CH_FL, CH_FR, CH_FC, CH_LFE, CH_BL, CH_BR],
        7 => vec![CH_FL, CH_FR, CH_FC, CH_LFE, CH_BC, CH_SL, CH_SR],
        8 => vec![CH_FL, CH_FR, CH_FC, CH_LFE, CH_BL, CH_BR, CH_SL, CH_SR],
        n => vec![-1; n],
    }
}

/// The fold to 5.1(side) by channel ROLE:
///
/// | input | output |
/// |---|---|
/// | FL FR FC LFE SL SR | the same channel, unchanged |
/// | BL BR, SL/SR also present | SL SR at 0.7071 |
/// | BL BR, no SL/SR (a 5.1(back) source) | SL SR at 1.0 |
/// | BC | SL and SR, 0.7071 each |
/// | any other role | the nearest output on its side, at 0.7071 |
///
/// "Nearest": front-half roles (FLC, wide, top/bottom front, stereo downmix L/R) go to FL/FR,
/// rear-half and side-height roles to SL/SR; centred heights split like BC (front ones into FC);
/// a second LFE joins the LFE. A role the lane cannot name (an unused or unknown slot) is dropped,
/// and outputs nothing feeds stay silent.
pub(super) fn fold_matrix(roles: &[i32]) -> Vec<FoldRow> {
    let has_side = roles.iter().any(|&r| r == CH_SL || r == CH_SR);
    let back_gain = if has_side { Q15_M3DB } else { Q15_ONE };
    roles
        .iter()
        .map(|&role| {
            let mut row = [0; OUT_CH];
            let mut put = |o: usize, g: i32| row[o] = g;
            match role {
                CH_FL => put(O_FL, Q15_ONE),
                CH_FR => put(O_FR, Q15_ONE),
                CH_FC => put(O_FC, Q15_ONE),
                CH_LFE => put(O_LFE, Q15_ONE),
                CH_SL => put(O_SL, Q15_ONE),
                CH_SR => put(O_SR, Q15_ONE),
                CH_BL => put(O_SL, back_gain),
                CH_BR => put(O_SR, back_gain),
                CH_BC | CH_TBC => {
                    put(O_SL, Q15_M3DB);
                    put(O_SR, Q15_M3DB);
                }
                CH_TC | CH_TFC | CH_BFC => put(O_FC, Q15_M3DB),
                CH_LFE2 => put(O_LFE, Q15_M3DB),
                CH_FLC | CH_TFL | CH_BFL | CH_WIDE_LEFT | CH_STEREO_LEFT => put(O_FL, Q15_M3DB),
                CH_FRC | CH_TFR | CH_BFR | CH_WIDE_RIGHT | CH_STEREO_RIGHT => put(O_FR, Q15_M3DB),
                CH_TBL | CH_SDL | CH_TSL | CH_SSL | CH_TSUL => put(O_SL, Q15_M3DB),
                CH_TBR | CH_SDR | CH_TSR | CH_SSR | CH_TSUR => put(O_SR, Q15_M3DB),
                _ => {}
            }
            row
        })
        .collect()
}

/// Interleaved full-scale S32 `input` (`matrix.len()` channels) folded to interleaved 5.1(side)
/// S32, appended to `out`. Accumulated in i64, rounded, and SATURATED to i32 — a full-scale BC
/// on top of a full-scale SL must clip, never wrap.
pub(super) fn fold(input: &[i32], matrix: &[FoldRow], out: &mut Vec<i32>) {
    let n = matrix.len();
    if n == 0 {
        return;
    }
    out.reserve(input.len() / n * OUT_CH);
    for frame in input.chunks_exact(n) {
        let mut acc = [0i64; OUT_CH];
        for (s, row) in frame.iter().zip(matrix) {
            for (a, g) in acc.iter_mut().zip(row) {
                *a += i64::from(*s) * i64::from(*g);
            }
        }
        for a in acc {
            let v = (a + (1 << 14)) >> 15;
            out.push(v.clamp(i64::from(i32::MIN), i64::from(i32::MAX)) as i32);
        }
    }
}

// ---- the FIFO -------------------------------------------------------------------------------

/// Folded 5.1 samples waiting to fill a 1536-sample frame, and the pts of the FIRST of them.
///
/// The first packet after open or seek anchors the clock; later packets only append. A packet
/// whose pts is more than ~100 ms off the running clock (anchor + samples held) is a
/// discontinuity — a demuxer gap, a lost packet run — so the partial frame is dropped and the
/// clock re-anchored on it, rather than stretching every later pts by the gap.
#[derive(Default)]
pub(super) struct Fifo {
    buf: Vec<i32>,
    anchor_ns: Option<i64>,
}

impl Fifo {
    pub(super) fn samples(&self) -> usize {
        self.buf.len() / OUT_CH
    }

    /// The pts the next appended sample will have, once anchored.
    pub(super) fn clock_ns(&self) -> Option<i64> {
        self.anchor_ns.map(|a| a + samples_ns(self.samples() as i64))
    }

    /// Append one packet's folded samples, whose first sample has `pts_ns` (None = the demuxer
    /// gave none: appended on the running clock). Returns true when it re-anchored. Samples that
    /// arrive before any pts are dropped — there is nothing to time them by.
    pub(super) fn push(&mut self, pts_ns: Option<i64>, folded: &[i32]) -> bool {
        let mut reanchored = false;
        match (self.clock_ns(), pts_ns) {
            (None, None) => return false,
            (None, Some(p)) => self.anchor_ns = Some(p),
            (Some(clock), Some(p)) if (p - clock).abs() > JUMP_NS => {
                self.buf.clear();
                self.anchor_ns = Some(p);
                reanchored = true;
            }
            _ => {}
        }
        self.buf.extend_from_slice(folded);
        reanchored
    }

    /// The next complete frame — `(pts_ns of its first sample, 1536 x 6 interleaved samples)` —
    /// or None while fewer than 1536 samples are held.
    pub(super) fn pop(&mut self) -> Option<(i64, Vec<i32>)> {
        let anchor = self.anchor_ns?;
        if self.samples() < AC3_FRAME {
            return None;
        }
        let frame: Vec<i32> = self.buf.drain(..AC3_FRAME * OUT_CH).collect();
        self.anchor_ns = Some(anchor + samples_ns(AC3_FRAME as i64));
        Some((anchor, frame))
    }

    /// After a seek: nothing held, nothing anchored.
    pub(super) fn clear(&mut self) {
        self.buf.clear();
        self.anchor_ns = None;
    }
}

/// The pts of the AC3 packet the encoder makes from the input frame starting at `frame_pts_ns`:
/// 256 samples earlier, for the encoder's priming.
pub(super) fn packet_pts(frame_pts_ns: i64) -> i64 {
    frame_pts_ns - samples_ns(AC3_PRIMING)
}

// ---- FFmpeg ---------------------------------------------------------------------------------

// AVSampleFormat values, asserted in ci/ffabi-assert.c.
const FMT_S16: c_int = 1;
const FMT_S32: c_int = 2;
const FMT_S16P: c_int = 6;
const FMT_S32P: c_int = 7;
/// AV_OPT_SEARCH_CHILDREN: reach a codec's PRIVATE options (TrueHD's `downmix`).
const OPT_SEARCH_CHILDREN: c_int = 1;
/// AV_CODEC_ID_AC3, asserted in ci/ffabi-assert.c.
const CODEC_ID_AC3: c_int = 0x15003;
/// AVERROR(EAGAIN) on Linux and the macOS host.
#[cfg(target_os = "macos")]
const ENC_EAGAIN: c_int = -35;
#[cfg(not(target_os = "macos"))]
const ENC_EAGAIN: c_int = -11;

// AVFrame fields the lane reads or writes, asserted in ci/ffabi-assert.c on both pointer widths.
const OFF_EFRAME_DATA: usize = 0;
#[cfg(target_pointer_width = "32")]
const OFF_EFRAME_NB_SAMPLES: usize = 76;
#[cfg(target_pointer_width = "32")]
const OFF_EFRAME_FORMAT: usize = 80;
#[cfg(target_pointer_width = "32")]
const OFF_EFRAME_SAMPLE_RATE: usize = 132;
#[cfg(target_pointer_width = "32")]
const OFF_EFRAME_CH_LAYOUT: usize = 256;
#[cfg(target_pointer_width = "64")]
const OFF_EFRAME_NB_SAMPLES: usize = 112;
#[cfg(target_pointer_width = "64")]
const OFF_EFRAME_FORMAT: usize = 116;
#[cfg(target_pointer_width = "64")]
const OFF_EFRAME_SAMPLE_RATE: usize = 180;
#[cfg(target_pointer_width = "64")]
const OFF_EFRAME_CH_LAYOUT: usize = 384;

#[inline]
unsafe fn frame_i32(frame: *const c_void, off: usize) -> i32 {
    *((frame as *const u8).add(off) as *const i32)
}

#[inline]
unsafe fn frame_set_i32(frame: *mut c_void, off: usize, v: i32) {
    *((frame as *mut u8).add(off) as *mut i32) = v;
}

#[inline]
unsafe fn frame_plane(frame: *const c_void, i: usize) -> *mut u8 {
    *((frame as *const u8).add(OFF_EFRAME_DATA) as *const *mut u8).add(i)
}

#[inline]
unsafe fn frame_layout(frame: *const c_void) -> *const AVChannelLayout {
    (frame as *const u8).add(OFF_EFRAME_CH_LAYOUT) as *const AVChannelLayout
}

/// The roles of a layout's channels, or [`default_roles`] when it states no order.
unsafe fn layout_roles(layout: *const AVChannelLayout) -> Vec<i32> {
    let n = (*layout).nb_channels.clamp(0, 64) as usize;
    let roles: Vec<i32> = (0..n).map(|i| av_channel_layout_channel_from_index(layout, i as c_uint)).collect();
    if roles.iter().all(|&r| r < 0) {
        default_roles(n)
    } else {
        roles
    }
}

fn cstr(s: &str) -> std::ffi::CString {
    std::ffi::CString::new(s).unwrap_or_default()
}

/// The source decoder (FLAC, TrueHD, MLP).
struct Decoder {
    ctx: *mut AVCodecContext,
    frame: *mut c_void,
}

impl Decoder {
    unsafe fn open(acp: *const AVCodecParameters, name: &str) -> Result<Decoder, String> {
        let codec = avcodec_find_decoder((*acp).codec_id);
        if codec.is_null() {
            return Err(format!("the bundled FFmpeg has no {name} decoder"));
        }
        let mut ctx = avcodec_alloc_context3(codec);
        if ctx.is_null() {
            return Err("avcodec_alloc_context3 failed".into());
        }
        if avcodec_parameters_to_context(ctx, acp) < 0 {
            avcodec_free_context(&mut ctx);
            return Err(format!("the {name} decoder refused the stream parameters"));
        }
        if matches!(name, "truehd" | "mlp") {
            // The stream's own 6-channel presentation when it carries one (TrueHD 7.1 does):
            // the decoder then reads one substream fewer and the fold has nothing to do.
            let r = av_opt_set(ctx as *mut c_void, cstr("downmix").as_ptr(), cstr("5.1(side)").as_ptr(), OPT_SEARCH_CHILDREN);
            if r < 0 {
                crate::player::log(&format!("ff: {name} downmix option refused ({r}); folding the full layout"));
            }
        }
        if avcodec_open2(ctx, codec, std::ptr::null_mut()) < 0 {
            avcodec_free_context(&mut ctx);
            return Err(format!("the {name} decoder refused the stream parameters"));
        }
        let frame = av_frame_alloc();
        if frame.is_null() {
            avcodec_free_context(&mut ctx);
            return Err("av_frame_alloc failed".into());
        }
        Ok(Decoder { ctx, frame })
    }
}

impl Drop for Decoder {
    fn drop(&mut self) {
        unsafe {
            av_frame_free(&mut self.frame);
            avcodec_free_context(&mut self.ctx);
        }
    }
}

/// `ac3_fixed`, configured by AVOption name (no AVCodecContext offset is read or written), with
/// one input frame allocated for the whole run.
struct Encoder {
    ctx: *mut AVCodecContext,
    frame: *mut c_void,
    pkt: *mut AVPacket,
    /// The pts of each frame sent and not yet returned as a packet.
    pending: std::collections::VecDeque<i64>,
}

impl Encoder {
    unsafe fn open() -> Result<Encoder, String> {
        let codec = avcodec_find_encoder_by_name(cstr("ac3_fixed").as_ptr());
        if codec.is_null() {
            return Err("the bundled FFmpeg has no ac3_fixed encoder".into());
        }
        let mut enc = Encoder {
            ctx: avcodec_alloc_context3(codec),
            frame: std::ptr::null_mut(),
            pkt: std::ptr::null_mut(),
            pending: Default::default(),
        };
        if enc.ctx.is_null() {
            return Err("avcodec_alloc_context3 failed".into());
        }
        // FFmpeg 9.0's AVCodecContext has no `sample_fmt` / `ch_layout` AVOption (only `ar`, `b`,
        // `time_base`), so the audio shape travels through a LIBRARY-ALLOCATED AVCodecParameters
        // whose fields are already modelled and asserted (format +28, ch_layout +112,
        // sample_rate +136 on ARM): `avcodec_parameters_to_context` copies format -> sample_fmt,
        // ch_layout and sample_rate. Still no AVCodecContext offset is read or written.
        let mut par = avcodec_parameters_alloc();
        if par.is_null() {
            return Err("avcodec_parameters_alloc failed".into());
        }
        (*par).codec_type = AVMEDIA_TYPE_AUDIO;
        (*par).codec_id = CODEC_ID_AC3;
        (*par).format = FMT_S32P;
        (*par).sample_rate = ENC_RATE as c_int;
        let shaped = av_channel_layout_from_string(&mut (*par).ch_layout, cstr("5.1(side)").as_ptr()) >= 0
            && (*par).ch_layout.nb_channels == OUT_CH as c_int
            && avcodec_parameters_to_context(enc.ctx, par) >= 0;
        avcodec_parameters_free(&mut par);
        if !shaped {
            return Err("ac3_fixed refused the S32P 5.1(side) 48 kHz parameters".into());
        }
        for (k, v) in [
            ("b", ENC_BITRATE.to_string()),
            ("ar", ENC_RATE.to_string()),
            ("time_base", format!("1/{ENC_RATE}")),
        ] {
            let r = av_opt_set(enc.ctx as *mut c_void, cstr(k).as_ptr(), cstr(&v).as_ptr(), 0);
            if r < 0 {
                return Err(format!("ac3_fixed refused {k}={v} ({r})"));
            }
        }
        let r = avcodec_open2(enc.ctx, codec, std::ptr::null_mut());
        if r < 0 {
            return Err(format!("ac3_fixed open failed ({r})"));
        }
        enc.frame = av_frame_alloc();
        enc.pkt = av_packet_alloc();
        if enc.frame.is_null() || enc.pkt.is_null() {
            return Err("frame/packet alloc failed".into());
        }
        frame_set_i32(enc.frame, OFF_EFRAME_NB_SAMPLES, AC3_FRAME as i32);
        frame_set_i32(enc.frame, OFF_EFRAME_FORMAT, FMT_S32P);
        frame_set_i32(enc.frame, OFF_EFRAME_SAMPLE_RATE, ENC_RATE as i32);
        let layout = (enc.frame as *mut u8).add(OFF_EFRAME_CH_LAYOUT) as *mut AVChannelLayout;
        if av_channel_layout_from_string(layout, cstr("5.1(side)").as_ptr()) < 0 || (*layout).nb_channels != OUT_CH as c_int {
            return Err("5.1(side) channel layout refused".into());
        }
        if av_frame_get_buffer(enc.frame, 0) < 0 {
            return Err("av_frame_get_buffer failed".into());
        }
        Ok(enc)
    }

    /// One 1536-sample interleaved 5.1 frame in; every packet the encoder returns appended to
    /// `out` with its pts. Err carries the libavcodec error.
    unsafe fn encode(&mut self, pts_ns: i64, frame: &[i32], out: &mut Vec<(i64, Vec<u8>)>) -> Result<(), c_int> {
        let r = av_frame_make_writable(self.frame);
        if r < 0 {
            return Err(r);
        }
        for c in 0..OUT_CH {
            let plane = frame_plane(self.frame, c) as *mut i32;
            if plane.is_null() {
                return Err(-1);
            }
            for i in 0..AC3_FRAME {
                *plane.add(i) = frame[i * OUT_CH + c];
            }
        }
        let r = avcodec_send_frame(self.ctx, self.frame);
        if r < 0 {
            return Err(r);
        }
        self.pending.push_back(pts_ns);
        loop {
            let r = avcodec_receive_packet(self.ctx, self.pkt);
            if r == ENC_EAGAIN || r == AVERROR_EOF {
                return Ok(());
            }
            if r < 0 {
                return Err(r);
            }
            let p = &*self.pkt;
            let bytes = if p.data.is_null() || p.size <= 0 {
                Vec::new()
            } else {
                std::slice::from_raw_parts(p.data, p.size as usize).to_vec()
            };
            let at = self.pending.pop_front().unwrap_or(pts_ns);
            av_packet_unref(self.pkt);
            if !bytes.is_empty() {
                out.push((packet_pts(at), bytes));
            }
        }
    }
}

impl Drop for Encoder {
    fn drop(&mut self) {
        unsafe {
            if !self.pkt.is_null() {
                av_packet_free(&mut self.pkt);
            }
            if !self.frame.is_null() {
                av_frame_free(&mut self.frame);
            }
            if !self.ctx.is_null() {
                avcodec_free_context(&mut self.ctx);
            }
        }
    }
}

enum Source {
    Decoded(Decoder),
    /// Native PCM: its sample layout and the channel roles the stream parameters state.
    Raw { fmt: SrcFmt, roles: Vec<i32> },
}

/// One audio stream fed through the encode lane.
pub(super) struct EncLane {
    src: Source,
    enc: Encoder,
    fifo: Fifo,
    /// The fold for the roles it was built from (a decoder may change layout mid-stream).
    roles: Vec<i32>,
    matrix: Vec<FoldRow>,
    s32: Vec<i32>,
    folded: Vec<i32>,
    out: Vec<(i64, Vec<u8>)>,
    errors: u32,
}

impl EncLane {
    /// The codecs the lane takes (FFmpeg names).
    pub(super) fn takes(name: &str) -> bool {
        matches!(name, "flac" | "truehd" | "mlp") || SrcFmt::from_pcm_codec(name).is_some()
    }

    /// Open the lane for the stream the Load declared as encode-lane AC3. `Err(reason)`: it
    /// cannot be fed as declared (the caller refuses the audio feed and says why).
    pub(super) unsafe fn open(acp: *const AVCodecParameters, spec: EncSpec) -> Result<EncLane, String> {
        let name = std::ffi::CStr::from_ptr(avcodec_get_name((*acp).codec_id)).to_string_lossy().into_owned();
        if !Self::takes(&name) {
            return Err(format!("{name} is not an encode-lane codec"));
        }
        let (channels, rate) = ((*acp).ch_layout.nb_channels, (*acp).sample_rate);
        if rate as u32 != ENC_RATE {
            return Err(format!("{name} {channels}ch at {rate} Hz: the encode lane takes {ENC_RATE} Hz only"));
        }
        if channels as u32 != spec.channels {
            crate::player::log(&format!(
                "ff: encode lane: {name} stream says {channels}ch, the Load declared {}ch (folding what the stream carries)",
                spec.channels
            ));
        }
        let src = match SrcFmt::from_pcm_codec(&name) {
            Some(fmt) => Source::Raw { fmt, roles: layout_roles(&(*acp).ch_layout) },
            None => Source::Decoded(Decoder::open(acp, &name)?),
        };
        let enc = Encoder::open()?;
        crate::player::log(&format!(
            "ff: audio lane {name} {channels}ch {rate} Hz decoded in-app -> AC3 5.1 {} kbps (encoded in-app)",
            ENC_BITRATE / 1000
        ));
        Ok(EncLane {
            src,
            enc,
            fifo: Fifo::default(),
            roles: Vec::new(),
            matrix: Vec::new(),
            s32: Vec::new(),
            folded: Vec::new(),
            out: Vec::new(),
            errors: 0,
        })
    }

    fn error(&mut self, what: &str) {
        self.errors += 1;
        if self.errors <= 3 {
            crate::player::log(&format!("ff: encode lane: {what}"));
        }
    }

    /// Fold `self.s32` (interleaved, `roles` channels) into the FIFO at `pts_ns`.
    fn take_samples(&mut self, roles: &[i32], pts_ns: Option<i64>) {
        if roles != self.roles.as_slice() {
            self.roles = roles.to_vec();
            self.matrix = fold_matrix(roles);
            crate::player::log(&format!("ff: encode lane folds roles {roles:?} to 5.1(side)"));
        }
        self.folded.clear();
        fold(&self.s32, &self.matrix, &mut self.folded);
        if self.fifo.push(pts_ns, &self.folded) {
            self.error("pts jump over 100 ms: partial frame dropped, clock re-anchored");
        }
    }

    /// One demuxed packet (first sample at `pts_ns`) to zero or more `(pts_ns, AC3 frame)`.
    pub(super) unsafe fn convert(&mut self, pkt: *mut AVPacket, pts_ns: Option<i64>) -> &[(i64, Vec<u8>)] {
        self.out.clear();
        let data: &[u8] = if (*pkt).data.is_null() || (*pkt).size <= 0 {
            &[]
        } else {
            std::slice::from_raw_parts((*pkt).data, (*pkt).size as usize)
        };
        if let Source::Raw { fmt, roles } = &self.src {
            let (fmt, roles) = (*fmt, roles.clone());
            self.s32.clear();
            let n = fmt.bytes();
            self.s32.extend(data.chunks_exact(n).map(|b| fmt.read(b)));
            let whole = self.s32.len() / roles.len().max(1) * roles.len();
            self.s32.truncate(whole);
            self.take_samples(&roles, pts_ns);
        } else {
            self.decode(pkt, pts_ns);
        }
        self.drain_frames();
        &self.out
    }

    unsafe fn decode(&mut self, pkt: *mut AVPacket, pts_ns: Option<i64>) {
        let Source::Decoded(dec) = &self.src else { return };
        let (ctx, frame) = (dec.ctx, dec.frame);
        let ret = avcodec_send_packet(ctx, pkt);
        if ret < 0 && ret != ENC_EAGAIN {
            self.error(&format!("decode error {ret} (packet skipped)"));
            return;
        }
        // Only the first frame of a packet carries the packet's pts; the rest follow on.
        let mut at = pts_ns;
        loop {
            let r = avcodec_receive_frame(ctx, frame);
            if r == ENC_EAGAIN || r == AVERROR_EOF {
                break;
            }
            if r < 0 {
                self.error(&format!("receive_frame error {r}"));
                break;
            }
            let samples = frame_i32(frame, OFF_EFRAME_NB_SAMPLES).max(0) as usize;
            let format = frame_i32(frame, OFF_EFRAME_FORMAT);
            let roles = layout_roles(frame_layout(frame));
            let ch = roles.len();
            self.s32.clear();
            match format {
                FMT_S16 | FMT_S32 => {
                    let p = frame_plane(frame, 0);
                    if !p.is_null() {
                        let w = if format == FMT_S16 { 2 } else { 4 };
                        let raw = std::slice::from_raw_parts(p, samples * ch * w);
                        let f = if format == FMT_S16 { SrcFmt::S16Le } else { SrcFmt::S32Le };
                        self.s32.extend(raw.chunks_exact(w).map(|b| f.read(b)));
                    }
                }
                FMT_S16P | FMT_S32P if ch <= 8 => {
                    let w = if format == FMT_S16P { 2 } else { 4 };
                    let f = if format == FMT_S16P { SrcFmt::S16Le } else { SrcFmt::S32Le };
                    let planes: Vec<*const u8> = (0..ch).map(|c| frame_plane(frame, c) as *const u8).collect();
                    if planes.iter().all(|p| !p.is_null()) {
                        self.s32.reserve(samples * ch);
                        for i in 0..samples {
                            for p in &planes {
                                self.s32.push(f.read(std::slice::from_raw_parts(p.add(i * w), w)));
                            }
                        }
                    }
                }
                other => {
                    self.error(&format!("decoder gave sample format {other} / {ch}ch; frame skipped"));
                    av_frame_unref(frame);
                    continue;
                }
            }
            av_frame_unref(frame);
            self.take_samples(&roles, at.take());
        }
    }

    /// Encode every complete frame the FIFO holds.
    unsafe fn drain_frames(&mut self) {
        while let Some((pts, frame)) = self.fifo.pop() {
            if let Err(r) = self.enc.encode(pts, &frame, &mut self.out) {
                self.error(&format!("ac3_fixed encode error {r} (frame skipped)"));
            }
        }
    }

    /// After a seek: no decoder, FIFO or encoder state may survive the discontinuity. The
    /// encoder is re-opened, because `ac3_fixed` keeps the previous block's samples for its
    /// MDCT overlap and has no flush of its own.
    pub(super) unsafe fn flush(&mut self) {
        if let Source::Decoded(dec) = &self.src {
            avcodec_flush_buffers(dec.ctx);
        }
        self.fifo.clear();
        match Encoder::open() {
            Ok(enc) => self.enc = enc,
            Err(why) => self.error(&format!("ac3_fixed re-open after seek failed: {why}")),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const L71: [i32; 8] = [CH_FL, CH_FR, CH_FC, CH_LFE, CH_BL, CH_BR, CH_SL, CH_SR];
    const L51_BACK: [i32; 6] = [CH_FL, CH_FR, CH_FC, CH_LFE, CH_BL, CH_BR];
    const L61: [i32; 7] = [CH_FL, CH_FR, CH_FC, CH_LFE, CH_BC, CH_SL, CH_SR];
    const L51_SIDE: [i32; 6] = [CH_FL, CH_FR, CH_FC, CH_LFE, CH_SL, CH_SR];

    fn row(pairs: &[(usize, i32)]) -> FoldRow {
        let mut r = [0; OUT_CH];
        for &(o, g) in pairs {
            r[o] = g;
        }
        r
    }

    #[test]
    fn the_fold_matrix_follows_channel_roles() {
        let one = Q15_ONE;
        let m3 = Q15_M3DB;
        // 7.1: the backs join the sides at -3 dB
        assert_eq!(
            fold_matrix(&L71),
            vec![
                row(&[(O_FL, one)]), row(&[(O_FR, one)]), row(&[(O_FC, one)]), row(&[(O_LFE, one)]),
                row(&[(O_SL, m3)]), row(&[(O_SR, m3)]), row(&[(O_SL, one)]), row(&[(O_SR, one)]),
            ]
        );
        // 5.1(back): with no sides, the backs ARE the surrounds
        assert_eq!(fold_matrix(&L51_BACK)[4], row(&[(O_SL, one)]));
        assert_eq!(fold_matrix(&L51_BACK)[5], row(&[(O_SR, one)]));
        // 6.1: the back centre splits into both sides
        assert_eq!(fold_matrix(&L61)[4], row(&[(O_SL, m3), (O_SR, m3)]));
        // 5.1(side): identity
        let id = fold_matrix(&L51_SIDE);
        for (i, r) in id.iter().enumerate() {
            assert_eq!(*r, row(&[(i, one)]), "slot {i}");
        }
        // others: nearest output on their side at -3 dB; unknown slots dropped
        let m = fold_matrix(&[CH_FLC, CH_FRC, CH_TBL, CH_TSR, CH_TC, CH_LFE2, -1, 0x300]);
        assert_eq!(m[0], row(&[(O_FL, m3)]));
        assert_eq!(m[1], row(&[(O_FR, m3)]));
        assert_eq!(m[2], row(&[(O_SL, m3)]));
        assert_eq!(m[3], row(&[(O_SR, m3)]));
        assert_eq!(m[4], row(&[(O_FC, m3)]));
        assert_eq!(m[5], row(&[(O_LFE, m3)]));
        assert_eq!(m[6], [0; OUT_CH]);
        assert_eq!(m[7], [0; OUT_CH]);
        // Q15 of 1/sqrt(2)
        assert!((f64::from(Q15_M3DB) / 32768.0 - std::f64::consts::FRAC_1_SQRT_2).abs() < 2e-5);
    }

    #[test]
    fn folding_moves_samples_and_saturates() {
        // 7.1 with a distinct value per channel
        let input: Vec<i32> = (1..=8).map(|v| v * 1_000_000).collect();
        let mut out = Vec::new();
        fold(&input, &fold_matrix(&L71), &mut out);
        let m3 = |v: i64| ((v * i64::from(Q15_M3DB) + (1 << 14)) >> 15) as i32;
        assert_eq!(out[..4], [1_000_000, 2_000_000, 3_000_000, 4_000_000]);
        assert_eq!(out[4], 7_000_000 + m3(5_000_000));
        assert_eq!(out[5], 8_000_000 + m3(6_000_000));
        // 5.1(side) passes through bit-exact, full scale included
        let edge = [i32::MAX, i32::MIN, 0, -1, 1, i32::MAX];
        let mut same = Vec::new();
        fold(&edge, &fold_matrix(&L51_SIDE), &mut same);
        assert_eq!(same, edge);
        // full-scale side + full-scale back centre clips at the rails, never wraps
        let loud = [0, 0, 0, 0, i32::MAX, i32::MAX, i32::MAX];
        let quiet = [0, 0, 0, 0, i32::MIN, i32::MIN, i32::MIN];
        let mut o = Vec::new();
        fold(&loud, &fold_matrix(&L61), &mut o);
        fold(&quiet, &fold_matrix(&L61), &mut o);
        assert_eq!(o[4..6], [i32::MAX, i32::MAX]);
        assert_eq!(o[10..12], [i32::MIN, i32::MIN]);
        // a trailing partial sample frame is ignored
        let mut p = Vec::new();
        fold(&[1, 2, 3], &fold_matrix(&[CH_FL, CH_FR]), &mut p);
        assert_eq!(p, [1, 2, 0, 0, 0, 0]);
    }

    #[test]
    fn unordered_layouts_take_ffmpegs_default_roles() {
        assert_eq!(default_roles(8), L71);
        assert_eq!(default_roles(6), L51_BACK);
        assert_eq!(default_roles(7), L61);
        assert_eq!(default_roles(2), [CH_FL, CH_FR]);
        assert_eq!(default_roles(10), vec![-1; 10]);
    }

    fn silence(samples: usize) -> Vec<i32> {
        vec![0; samples * OUT_CH]
    }

    #[test]
    fn the_fifo_cuts_exact_frames_on_the_anchored_clock() {
        assert_eq!(samples_ns(AC3_FRAME as i64), 32_000_000);
        assert_eq!(packet_pts(1_000_000_000), 1_000_000_000 - 5_333_333);
        let mut f = Fifo::default();
        // nothing to time samples by before the first pts
        assert!(!f.push(None, &silence(100)));
        assert_eq!(f.samples(), 0);
        // the first packet anchors; later ones only append (here 1000-sample packets)
        let t0 = 10_000_000_000;
        assert!(!f.push(Some(t0), &silence(1000)));
        assert!(f.pop().is_none());
        assert!(!f.push(Some(t0 + samples_ns(1000)), &silence(1000)));
        assert_eq!(f.pop().map(|(p, fr)| (p, fr.len())), Some((t0, AC3_FRAME * OUT_CH)));
        assert!(f.pop().is_none());
        assert_eq!(f.samples(), 2000 - AC3_FRAME);
        // a packet without pts rides the running clock
        assert!(!f.push(None, &silence(1072)));
        assert_eq!(f.pop().map(|(p, _)| p), Some(t0 + 32_000_000));
        assert_eq!(f.clock_ns(), Some(t0 + samples_ns(3072)));
        // small jitter is not a jump
        assert!(!f.push(Some(t0 + samples_ns(3072) + 20_000_000), &silence(10)));
        assert_eq!(f.samples(), 10);
        // a jump over 100 ms drops the partial frame and re-anchors on the new pts
        let t1 = t0 + 5_000_000_000;
        assert!(f.push(Some(t1), &silence(1536)));
        assert_eq!(f.pop().map(|(p, _)| p), Some(t1));
        assert_eq!(f.samples(), 0);
        // flush (seek): nothing held, the next packet anchors afresh — backwards is fine too
        f.push(Some(t1 + 32_000_000), &silence(700));
        f.clear();
        assert_eq!((f.samples(), f.clock_ns()), (0, None));
        assert!(!f.push(Some(2_000_000_000), &silence(1536)));
        assert_eq!(f.pop().map(|(p, _)| packet_pts(p)), Some(2_000_000_000 - 5_333_333));
    }

    /// **Round trip through the bundled FFmpeg** (`ac3_fixed`): a synthetic 7.1 S32 sine fed as
    /// native PCM through the whole lane — fold, FIFO, encoder — must come out as one AC3 frame
    /// per 1536 samples, each 2560 bytes (640 kbps at 48 kHz) behind the 0x0B77 sync word, on a
    /// 32 ms pts step. Needs the host build of the bundled FFmpeg, like the other FFmpeg tests.
    #[test]
    #[ignore = "needs the host build of the bundled FFmpeg — run by `make check-ffmpeg`"]
    fn a_71_sine_round_trips_to_640_kbps_ac3() {
        let _g = plx_base::testlock::serial();
        super::super::image_subtitle_tests::bind_host_ffmpeg();
        let mut lane = unsafe {
            EncLane {
                src: Source::Raw { fmt: SrcFmt::S32Le, roles: L71.to_vec() },
                enc: Encoder::open().expect("ac3_fixed opens"),
                fifo: Fifo::default(),
                roles: Vec::new(),
                matrix: Vec::new(),
                s32: Vec::new(),
                folded: Vec::new(),
                out: Vec::new(),
                errors: 0,
            }
        };
        let total = 48_000usize; // one second
        let per_packet = 1000usize;
        let t0 = 3_000_000_000i64;
        let mut got: Vec<(i64, Vec<u8>)> = Vec::new();
        let mut buf = Vec::new();
        for start in (0..total).step_by(per_packet) {
            buf.clear();
            for i in start..start + per_packet {
                for c in 0..8 {
                    // An integer triangle wave per channel, period 48000 / (220 * (c + 1))
                    // samples at half scale: no libm call (`ci/allow/libm.txt` only shrinks).
                    let period = (48_000 / (220 * (c + 1))) as i64;
                    let (p, half) = ((i as i64) % period, period / 2);
                    let tri = if p < half { p } else { period - p };
                    let v = ((tri * 2 - half) * (i64::from(i32::MAX) / 2) / half.max(1)) as i32;
                    buf.extend_from_slice(&v.to_le_bytes());
                }
            }
            let mut pkt: AVPacket = unsafe { std::mem::zeroed() };
            pkt.data = buf.as_mut_ptr();
            pkt.size = buf.len() as c_int;
            let pts = t0 + samples_ns(start as i64);
            let out = unsafe { lane.convert(&mut pkt, Some(pts)) };
            got.extend(out.iter().cloned());
        }
        assert_eq!(lane.errors, 0);
        assert_eq!(got.len(), total / AC3_FRAME, "one packet per 1536 samples");
        for (k, (pts, bytes)) in got.iter().enumerate() {
            assert_eq!(&bytes[..2], &[0x0b, 0x77], "packet {k}: AC3 sync word");
            assert_eq!(bytes.len(), 2560, "packet {k}: 640 kbps frame size");
            assert_eq!(*pts, packet_pts(t0 + 32_000_000 * k as i64), "packet {k}: 32 ms step");
        }
    }

    /// FORK (RichiL10): **real FLAC and TrueHD streams through the whole lane**: decoder (and
    /// TrueHD's `downmix` option), the layout read back off the decoded frame, fold, FIFO and
    /// encoder. The bundled FFmpeg has no FLAC/TrueHD encoder, so a full FFmpeg makes the fixtures
    /// into `PLX_ENC_FIXTURES`: 3 s of 48 kHz tones, one per channel, in Matroska - `flac71.mkv`
    /// (24-bit 7.1), `flac51.mkv` (24-bit 5.1(side)) and `truehd.mkv` (the widest layout the
    /// experimental TrueHD encoder accepts). A fixture that was not generated is skipped by name.
    #[test]
    #[ignore = "needs the host build of the bundled FFmpeg and PLX_ENC_FIXTURES"]
    fn real_flac_and_truehd_streams_encode_to_ac3() {
        let _g = plx_base::testlock::serial();
        super::super::image_subtitle_tests::bind_host_ffmpeg();
        let dir = std::path::PathBuf::from(std::env::var_os("PLX_ENC_FIXTURES").expect("PLX_ENC_FIXTURES"));
        let mut ran = 0;
        for name in ["flac71.mkv", "flac51.mkv", "truehd.mkv"] {
            let path = dir.join(name);
            if !path.exists() {
                eprintln!("skip {name}: not generated");
                continue;
            }
            ran += 1;
            let url = std::ffi::CString::new(path.to_str().unwrap()).unwrap();
            unsafe {
                let mut fmt: *mut AVFormatContext = std::ptr::null_mut();
                assert!(
                    avformat_open_input(&mut fmt, url.as_ptr(), std::ptr::null_mut(), std::ptr::null_mut()) >= 0,
                    "{name}: open"
                );
                assert!(avformat_find_stream_info(fmt, std::ptr::null_mut()) >= 0, "{name}: stream info");
                let streams = (*fmt).streams;
                let ai = (0..(*fmt).nb_streams as usize)
                    .find(|&i| (*stream_codecpar(*streams.add(i))).codec_type == AVMEDIA_TYPE_AUDIO)
                    .expect("an audio stream");
                let st = *streams.add(ai);
                let acp = stream_codecpar(st);
                let channels = (*acp).ch_layout.nb_channels as u32;
                let mut lane = EncLane::open(acp, EncSpec { channels }).unwrap_or_else(|e| panic!("{name}: {e}"));
                let mut pkt = av_packet_alloc();
                let mut got: Vec<(i64, Vec<u8>)> = Vec::new();
                while av_read_frame(fmt, pkt) >= 0 {
                    if (*pkt).stream_index as usize == ai {
                        let pts = pts_ns_opt(pkt, st);
                        got.extend(lane.convert(pkt, pts).iter().cloned());
                    }
                    av_packet_unref(pkt);
                }
                av_packet_free(&mut pkt);
                avformat_close_input(&mut fmt);
                assert_eq!(lane.errors, 0, "{name}: lane errors");
                // 3 s at 48 kHz = 93.75 frames of 1536 samples; the partial last one stays queued.
                assert!((90..=94).contains(&got.len()), "{name}: {} AC3 frames", got.len());
                for (k, (_, bytes)) in got.iter().enumerate() {
                    assert_eq!(&bytes[..2], &[0x0b, 0x77], "{name} packet {k}: AC3 sync word");
                    assert_eq!(bytes.len(), 2560, "{name} packet {k}: 640 kbps frame size");
                }
                for w in got.windows(2) {
                    assert_eq!(w[1].0 - w[0].0, 32_000_000, "{name}: 32 ms pts step");
                }
                // tones went in, so the frames cannot all be the same bytes
                assert!(got.windows(2).any(|w| w[0].1 != w[1].1), "{name}: every AC3 frame identical");
                // PLX_ENC_DUMP=1: keep the elementary AC3 stream beside the fixture, so a full
                // FFmpeg can decode it and check each output channel's tones (the channel mapping).
                if std::env::var_os("PLX_ENC_DUMP").is_some() {
                    let es: Vec<u8> = got.iter().flat_map(|(_, b)| b.iter().copied()).collect();
                    std::fs::write(dir.join(format!("out-{name}.ac3")), es).expect("dump the AC3 stream");
                }
                eprintln!("{name}: {channels}ch -> {} AC3 frames", got.len());
            }
        }
        assert!(ran > 0, "no fixture in PLX_ENC_FIXTURES");
    }
}
