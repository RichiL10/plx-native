//! FORK (RichiL10): the PCM audio lane, and the declarations the Load makes for the fork's
//! audio lanes.
//!
//! LG's buffer-feed pipeline accepts `codec.audio = "PCM"` with a `contents.pcmInfo` block
//! (format, channelMode, layout, sampleRate, bitsPerSample: read off `libpf-1.0.so` on the B4,
//! and the same block moonlight-tv's webOS driver plays with). The fork feeds two kinds of
//! source through it:
//!
//! * **FLAC**, decoded HERE with the bundled FFmpeg's `flac` decoder (`ci/build-ffmpeg.sh`). It
//!   leaves the app bit-exact: the decoder's MSB-aligned S32 output is written at the declared
//!   width, so a 24-bit source declared S32LE or S24LE loses nothing.
//! * **Native PCM tracks** (Matroska `A_PCM/*`), converted to the declared width/endianness.
//!
//! The Load payload is composed from the ROUTE's metadata before this demuxer opens the stream,
//! so the format is declared first ([`PcmSpec`], published by the engine through [`declare`])
//! and this lane's whole job is to emit exactly that: the source's own rate and channel count
//! (nothing here resamples or remixes; a stream that disagrees with its declaration is refused
//! loudly instead of being played at the wrong pitch or with the wrong speakers) at the declared
//! sample width.

use super::*;

/// The sample width the lane emits — always interleaved and little-endian.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PcmWidth {
    S16,
    S24,
    S32,
}

impl PcmWidth {
    pub fn bytes(self) -> usize {
        match self {
            PcmWidth::S16 => 2,
            PcmWidth::S24 => 3,
            PcmWidth::S32 => 4,
        }
    }
    /// `pcmInfo.format` — spellings from libpf's own table (S16LE, S24LE, S32LE).
    pub fn format(self) -> &'static str {
        match self {
            PcmWidth::S16 => "S16LE",
            PcmWidth::S24 => "S24LE",
            PcmWidth::S32 => "S32LE",
        }
    }
    pub fn bits(self) -> u32 {
        self.bytes() as u32 * 8
    }
}

/// What the Load declared for the PCM lane.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PcmSpec {
    pub rate: u32,
    pub channels: u32,
    pub width: PcmWidth,
}

impl PcmSpec {
    /// The declaration for a PCM-lane track from the route's metadata (PMS `samplingRate`,
    /// `channels`, `bitDepth`). Sources up to 16 bit are declared S16LE (exact); deeper or
    /// unknown depths follow the `pcm_hi` knob: `s32` (default), `s24`, or `s16`. `None` when
    /// the rate is unknown or the channel count is not one the lane can declare.
    pub fn for_track(rate: i64, channels: i64, bit_depth: i64, pcm_hi: Option<&str>) -> Option<PcmSpec> {
        if rate <= 0 || !plx_platform::devcaps::pcm_lane_channels_ok(channels) {
            return None;
        }
        let width = if (1..=16).contains(&bit_depth) {
            PcmWidth::S16
        } else {
            match pcm_hi {
                Some("s16") => PcmWidth::S16,
                Some("s24") => PcmWidth::S24,
                _ => PcmWidth::S32,
            }
        };
        Some(PcmSpec {
            rate: u32::try_from(rate).ok()?,
            channels: channels as u32,
            width,
        })
    }

    /// `pcmInfo.channelMode`.
    pub fn channel_mode(self) -> &'static str {
        match self.channels {
            1 => "mono",
            6 => "6-channel",
            _ => "stereo",
        }
    }
}

/// The Load's PCM declaration for the session being started; `None` when the Load declared
/// another codec. Written by the engine BEFORE the demux thread exists, read once by it at open.
static DECLARED: std::sync::Mutex<Option<PcmSpec>> = std::sync::Mutex::new(None);

pub fn declare(spec: Option<PcmSpec>) {
    *DECLARED.lock().unwrap_or_else(|e| e.into_inner()) = spec;
}

pub(super) fn declared() -> Option<PcmSpec> {
    *DECLARED.lock().unwrap_or_else(|e| e.into_inner())
}

/// A source sample layout the lane reads.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum SrcFmt {
    S16Le,
    S16Be,
    S24Le,
    S24Be,
    S32Le,
    S32Be,
    F32Le,
    F32Be,
}

impl SrcFmt {
    fn bytes(self) -> usize {
        match self {
            SrcFmt::S16Le | SrcFmt::S16Be => 2,
            SrcFmt::S24Le | SrcFmt::S24Be => 3,
            _ => 4,
        }
    }

    /// The Matroska/MOV PCM flavours FFmpeg names (`avcodec_get_name`, stable across majors).
    pub(super) fn from_pcm_codec(name: &str) -> Option<SrcFmt> {
        Some(match name {
            "pcm_s16le" => SrcFmt::S16Le,
            "pcm_s16be" => SrcFmt::S16Be,
            "pcm_s24le" => SrcFmt::S24Le,
            "pcm_s24be" => SrcFmt::S24Be,
            "pcm_s32le" => SrcFmt::S32Le,
            "pcm_s32be" => SrcFmt::S32Be,
            "pcm_f32le" => SrcFmt::F32Le,
            "pcm_f32be" => SrcFmt::F32Be,
            _ => return None,
        })
    }

    /// One sample as a full-scale (MSB-aligned) 32-bit value.
    #[inline]
    fn read(self, b: &[u8]) -> i32 {
        match self {
            SrcFmt::S16Le => i32::from(i16::from_le_bytes([b[0], b[1]])) << 16,
            SrcFmt::S16Be => i32::from(i16::from_be_bytes([b[0], b[1]])) << 16,
            SrcFmt::S24Le => i32::from_le_bytes([0, b[0], b[1], b[2]]),
            SrcFmt::S24Be => i32::from_be_bytes([b[0], b[1], b[2], 0]),
            SrcFmt::S32Le => i32::from_le_bytes([b[0], b[1], b[2], b[3]]),
            SrcFmt::S32Be => i32::from_be_bytes([b[0], b[1], b[2], b[3]]),
            SrcFmt::F32Le => float_to_s32(f32::from_le_bytes([b[0], b[1], b[2], b[3]])),
            SrcFmt::F32Be => float_to_s32(f32::from_be_bytes([b[0], b[1], b[2], b[3]])),
        }
    }
}

#[inline]
fn float_to_s32(f: f32) -> i32 {
    if f.is_nan() {
        return 0;
    }
    let v = f64::from(f.clamp(-1.0, 1.0)) * 2_147_483_647.0;
    v.round() as i32
}

/// One full-scale sample written at `out`'s width, little-endian. Narrowing truncates, which is
/// exact for a source no deeper than the target (the only case the declaration produces unless
/// the `pcm_hi=s16` knob deliberately asks for 16 bit).
#[inline]
fn write(v: i32, out: PcmWidth, dst: &mut Vec<u8>) {
    let b = v.to_le_bytes();
    match out {
        PcmWidth::S16 => dst.extend_from_slice(&b[2..4]),
        PcmWidth::S24 => dst.extend_from_slice(&b[1..4]),
        PcmWidth::S32 => dst.extend_from_slice(&b),
    }
}

/// Interleaved source samples to interleaved little-endian `out` samples, appended to `dst`.
/// A trailing partial sample is ignored.
pub(super) fn convert_interleaved(src: &[u8], fmt: SrcFmt, out: PcmWidth, dst: &mut Vec<u8>) {
    let n = fmt.bytes();
    dst.reserve(src.len() / n * out.bytes());
    for s in src.chunks_exact(n) {
        write(fmt.read(s), out, dst);
    }
}

/// Planar source samples (one slice per channel, `samples` each) interleaved into `dst`.
pub(super) fn convert_planar(planes: &[&[u8]], samples: usize, fmt: SrcFmt, out: PcmWidth, dst: &mut Vec<u8>) {
    let n = fmt.bytes();
    dst.reserve(samples * planes.len() * out.bytes());
    for i in 0..samples {
        for p in planes {
            if let Some(s) = p.get(i * n..i * n + n) {
                write(fmt.read(s), out, dst);
            }
        }
    }
}

// AVSampleFormat values (libavutil/samplefmt.h; unchanged since FFmpeg 1.0, and asserted in
// ci/ffabi-assert.c for the bundled 9.0).
const AV_SAMPLE_FMT_S16: c_int = 1;
const AV_SAMPLE_FMT_S32: c_int = 2;
const AV_SAMPLE_FMT_S16P: c_int = 6;
const AV_SAMPLE_FMT_S32P: c_int = 7;
/// AVERROR(EAGAIN) on Linux and the macOS host (EAGAIN is 11 on Linux, 35 on Darwin).
#[cfg(target_os = "macos")]
const AVERROR_EAGAIN: c_int = -35;
#[cfg(not(target_os = "macos"))]
const AVERROR_EAGAIN: c_int = -11;

// AVFrame fields the decode path reads, beside the encoder's OFF_FRAME_* table and asserted in
// ci/ffabi-assert.c on both pointer widths: `data` (+0), `nb_samples` (+76 / +112) and `format`
// (+80 / +116).
const OFF_AFRAME_DATA: usize = 0;
#[cfg(target_pointer_width = "32")]
const OFF_AFRAME_NB_SAMPLES: usize = 76;
#[cfg(target_pointer_width = "32")]
const OFF_AFRAME_FORMAT: usize = 80;
#[cfg(target_pointer_width = "64")]
const OFF_AFRAME_NB_SAMPLES: usize = 112;
#[cfg(target_pointer_width = "64")]
const OFF_AFRAME_FORMAT: usize = 116;

#[inline]
unsafe fn frame_i32(frame: *const c_void, off: usize) -> i32 {
    *((frame as *const u8).add(off) as *const i32)
}

#[inline]
unsafe fn frame_plane(frame: *const c_void, i: usize) -> *const u8 {
    *((frame as *const u8).add(OFF_AFRAME_DATA) as *const *const u8).add(i)
}

/// The bundled FFmpeg's FLAC decoder, owned for one demux run.
struct Decoder {
    ctx: *mut AVCodecContext,
    frame: *mut c_void,
}

impl Decoder {
    unsafe fn open(acp: *const AVCodecParameters) -> Result<Decoder, String> {
        let codec = avcodec_find_decoder((*acp).codec_id);
        if codec.is_null() {
            return Err("the bundled FFmpeg has no flac decoder".into());
        }
        let mut ctx = avcodec_alloc_context3(codec);
        if ctx.is_null() {
            return Err("avcodec_alloc_context3 failed".into());
        }
        if avcodec_parameters_to_context(ctx, acp) < 0 || avcodec_open2(ctx, codec, std::ptr::null_mut()) < 0 {
            avcodec_free_context(&mut ctx);
            return Err("the flac decoder refused the stream parameters".into());
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

enum Source {
    Flac(Decoder),
    Raw(SrcFmt),
}

/// One audio stream fed through the PCM lane.
pub(super) struct PcmLane {
    src: Source,
    spec: PcmSpec,
    buf: Vec<u8>,
    errors: u32,
}

impl PcmLane {
    /// `Ok(None)`: the stream is not a PCM-lane codec. `Err(reason)`: it is one, but cannot be
    /// fed as the Load declared it (the caller refuses the audio feed and says why).
    pub(super) unsafe fn open(acp: *const AVCodecParameters, spec: Option<PcmSpec>) -> Result<Option<PcmLane>, String> {
        let name = std::ffi::CStr::from_ptr(avcodec_get_name((*acp).codec_id)).to_string_lossy().into_owned();
        let raw = SrcFmt::from_pcm_codec(&name);
        if name != "flac" && raw.is_none() {
            return Ok(None);
        }
        let Some(spec) = spec else {
            return Err(format!("{name}: the Load declared no PCM format for it"));
        };
        let (channels, rate) = ((*acp).ch_layout.nb_channels, (*acp).sample_rate);
        if channels as u32 != spec.channels || rate as u32 != spec.rate {
            return Err(format!(
                "{name} {channels}ch {rate} Hz does not match the declared {}ch {} Hz",
                spec.channels, spec.rate
            ));
        }
        let src = match raw {
            Some(fmt) => Source::Raw(fmt),
            None => Source::Flac(Decoder::open(acp)?),
        };
        let what = match &src {
            Source::Flac(_) => "FLAC decoded in-app".to_string(),
            Source::Raw(_) => format!("PCM {name}"),
        };
        crate::player::log(&format!(
            "ff: audio lane {what} -> PCM {} {}ch {} Hz",
            spec.width.format(),
            spec.channels,
            spec.rate
        ));
        Ok(Some(PcmLane { src, spec, buf: Vec::new(), errors: 0 }))
    }

    /// One demuxed packet to the declared PCM. Empty when the packet produced no samples (a
    /// decode error, logged a few times then counted silently).
    pub(super) unsafe fn convert(&mut self, pkt: *const AVPacket) -> &[u8] {
        self.buf.clear();
        let out = self.spec.width;
        let data: &[u8] = if (*pkt).data.is_null() || (*pkt).size <= 0 {
            &[]
        } else {
            std::slice::from_raw_parts((*pkt).data, (*pkt).size as usize)
        };
        match &mut self.src {
            Source::Raw(fmt) => convert_interleaved(data, *fmt, out, &mut self.buf),
            Source::Flac(dec) => {
                let ret = avcodec_send_packet(dec.ctx, pkt);
                if ret < 0 && ret != AVERROR_EAGAIN {
                    self.errors += 1;
                    if self.errors <= 3 {
                        crate::player::log(&format!("ff: flac decode error {ret} (packet skipped)"));
                    }
                    return &self.buf;
                }
                let channels = self.spec.channels as usize;
                loop {
                    let r = avcodec_receive_frame(dec.ctx, dec.frame);
                    if r == AVERROR_EAGAIN || r == AVERROR_EOF {
                        break;
                    }
                    if r < 0 {
                        self.errors += 1;
                        if self.errors <= 3 {
                            crate::player::log(&format!("ff: flac receive_frame error {r}"));
                        }
                        break;
                    }
                    let samples = frame_i32(dec.frame, OFF_AFRAME_NB_SAMPLES).max(0) as usize;
                    let format = frame_i32(dec.frame, OFF_AFRAME_FORMAT);
                    let (fmt, planar) = match format {
                        AV_SAMPLE_FMT_S16 => (SrcFmt::S16Le, false),
                        AV_SAMPLE_FMT_S32 => (SrcFmt::S32Le, false),
                        AV_SAMPLE_FMT_S16P => (SrcFmt::S16Le, true),
                        AV_SAMPLE_FMT_S32P => (SrcFmt::S32Le, true),
                        other => {
                            self.errors += 1;
                            if self.errors <= 3 {
                                crate::player::log(&format!("ff: flac decoder gave sample format {other}; frame skipped"));
                            }
                            av_frame_unref(dec.frame);
                            continue;
                        }
                    };
                    let n = fmt.bytes();
                    if planar {
                        let planes: Vec<&[u8]> = (0..channels.min(8))
                            .map(|c| {
                                let p = frame_plane(dec.frame, c);
                                if p.is_null() { &[][..] } else { std::slice::from_raw_parts(p, samples * n) }
                            })
                            .collect();
                        convert_planar(&planes, samples, fmt, out, &mut self.buf);
                    } else {
                        let p = frame_plane(dec.frame, 0);
                        if !p.is_null() {
                            let src = std::slice::from_raw_parts(p, samples * channels * n);
                            convert_interleaved(src, fmt, out, &mut self.buf);
                        }
                    }
                    av_frame_unref(dec.frame);
                }
            }
        }
        &self.buf
    }

    /// After a seek: the decoder must not carry state across the discontinuity.
    pub(super) unsafe fn flush(&mut self) {
        if let Source::Flac(dec) = &mut self.src {
            avcodec_flush_buffers(dec.ctx);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spec_from_track_metadata() {
        let s = PcmSpec::for_track(48_000, 2, 24, None).unwrap();
        assert_eq!((s.rate, s.channels, s.width), (48_000, 2, PcmWidth::S32));
        assert_eq!(s.channel_mode(), "stereo");
        assert_eq!(PcmSpec::for_track(44_100, 2, 16, None).unwrap().width, PcmWidth::S16);
        assert_eq!(PcmSpec::for_track(48_000, 2, 0, None).unwrap().width, PcmWidth::S32);
        assert_eq!(PcmSpec::for_track(48_000, 2, 24, Some("s24")).unwrap().width, PcmWidth::S24);
        assert_eq!(PcmSpec::for_track(48_000, 2, 24, Some("s16")).unwrap().width, PcmWidth::S16);
        assert_eq!(PcmSpec::for_track(48_000, 1, 16, None).unwrap().channel_mode(), "mono");
        assert_eq!(PcmSpec::for_track(48_000, 6, 24, None).unwrap().channel_mode(), "6-channel");
        assert!(PcmSpec::for_track(0, 2, 16, None).is_none());
        assert!(PcmSpec::for_track(48_000, 8, 24, None).is_none());
        assert!(PcmSpec::for_track(48_000, 0, 24, None).is_none());
    }

    #[test]
    fn flac_s32_msb_aligned_output_is_written_bit_exact() {
        // FFmpeg's flac decoder hands 24-bit audio over as S32 shifted up by 8.
        let samples: [i32; 4] = [0x7fff_ff00, i32::MIN, 0x0012_3400, -0x0000_0100];
        let src: Vec<u8> = samples.iter().flat_map(|v| v.to_le_bytes()).collect();
        let mut s32 = Vec::new();
        convert_interleaved(&src, SrcFmt::S32Le, PcmWidth::S32, &mut s32);
        assert_eq!(s32, src);
        let mut s24 = Vec::new();
        convert_interleaved(&src, SrcFmt::S32Le, PcmWidth::S24, &mut s24);
        assert_eq!(s24, [0xff, 0xff, 0x7f, 0x00, 0x00, 0x80, 0x34, 0x12, 0x00, 0xff, 0xff, 0xff]);
        let mut s16 = Vec::new();
        convert_interleaved(&src, SrcFmt::S32Le, PcmWidth::S16, &mut s16);
        assert_eq!(s16, [0xff, 0x7f, 0x00, 0x80, 0x12, 0x00, 0xff, 0xff]);
    }

    #[test]
    fn sixteen_bit_sources_round_trip_at_every_width() {
        let samples: [i16; 3] = [1, -2, i16::MAX];
        let le: Vec<u8> = samples.iter().flat_map(|v| v.to_le_bytes()).collect();
        let be: Vec<u8> = samples.iter().flat_map(|v| v.to_be_bytes()).collect();
        let mut a = Vec::new();
        convert_interleaved(&le, SrcFmt::S16Le, PcmWidth::S16, &mut a);
        assert_eq!(a, le);
        let mut b = Vec::new();
        convert_interleaved(&be, SrcFmt::S16Be, PcmWidth::S16, &mut b);
        assert_eq!(b, le);
        let mut c = Vec::new();
        convert_interleaved(&le, SrcFmt::S16Le, PcmWidth::S32, &mut c);
        let back: Vec<i32> = c.chunks_exact(4).map(|x| i32::from_le_bytes([x[0], x[1], x[2], x[3]])).collect();
        assert_eq!(back, [1 << 16, -2 << 16, i32::from(i16::MAX) << 16]);
    }

    #[test]
    fn packed_24_bit_and_float_sources() {
        let s24le = [0x56, 0x34, 0x12, 0xff, 0xff, 0xff];
        let mut out = Vec::new();
        convert_interleaved(&s24le, SrcFmt::S24Le, PcmWidth::S24, &mut out);
        assert_eq!(out, s24le);
        let s24be = [0x12, 0x34, 0x56, 0xff, 0xff, 0xff];
        let mut out = Vec::new();
        convert_interleaved(&s24be, SrcFmt::S24Be, PcmWidth::S24, &mut out);
        assert_eq!(out, s24le);
        let f: Vec<u8> = [1.0f32, -1.0, 0.0, 2.0].iter().flat_map(|v| v.to_le_bytes()).collect();
        let mut out = Vec::new();
        convert_interleaved(&f, SrcFmt::F32Le, PcmWidth::S16, &mut out);
        assert_eq!(out, [0xff, 0x7f, 0x00, 0x80, 0x00, 0x00, 0xff, 0x7f]);
    }

    #[test]
    fn planar_frames_are_interleaved() {
        let l: Vec<u8> = [1i16, 3].iter().flat_map(|v| v.to_le_bytes()).collect();
        let r: Vec<u8> = [2i16, 4].iter().flat_map(|v| v.to_le_bytes()).collect();
        let mut out = Vec::new();
        convert_planar(&[&l, &r], 2, SrcFmt::S16Le, PcmWidth::S16, &mut out);
        let got: Vec<i16> = out.chunks_exact(2).map(|x| i16::from_le_bytes([x[0], x[1]])).collect();
        assert_eq!(got, [1, 2, 3, 4]);
    }

    #[test]
    fn pcm_codec_names_map_to_layouts() {
        assert_eq!(SrcFmt::from_pcm_codec("pcm_s24le"), Some(SrcFmt::S24Le));
        assert_eq!(SrcFmt::from_pcm_codec("pcm_s16be"), Some(SrcFmt::S16Be));
        assert_eq!(SrcFmt::from_pcm_codec("flac"), None);
        assert_eq!(SrcFmt::from_pcm_codec("pcm_bluray"), None);
    }
}
