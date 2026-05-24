import React, { useState, useRef, useEffect, useCallback } from "react";
import WaveSurfer from "wavesurfer.js";
import { FaGuitar, FaDrum, FaMicrophone, FaMusic, FaPlay, FaPause } from "react-icons/fa";
import { GiFlute, GiTrumpet, GiSaxophone, GiViolin } from "react-icons/gi";
import { LuPiano } from "react-icons/lu";
import { createNoise3D } from "simplex-noise";

const API_BASE = import.meta.env.VITE_API_URL || "";

const DISPLAY_NAMES = {
  bass: "Bass Guitar",
  cello: "Cello",
  clarinet: "Clarinet",
  drums: "Drums",
  flute: "Flute",
  gac: "Acoustic Guitar",
  gel: "Electric Guitar",
  organ: "Organ",
  piano: "Piano",
  saxophone: "Saxophone",
  trumpet: "Trumpet",
  violin: "Violin",
  voice: "Voice",
};

const INSTRUMENT_ICONS = {
  bass: <FaGuitar />,
  cello: <GiViolin />,
  clarinet: <GiFlute />,
  flute: <GiFlute />,
  drums: <FaDrum />,
  gac: <FaGuitar />,
  gel: <FaGuitar />,
  organ: <LuPiano />,
  piano: <LuPiano />,
  saxophone: <GiSaxophone />,
  trumpet: <GiTrumpet />,
  violin: <GiViolin />,
  voice: <FaMicrophone />,
};

function getRole(score) {
  if (score >= 0.65) return { label: "Lead",       className: "bg-[#00c8ff]/20 text-[#00c8ff] border-[#00c8ff]/40" };
  if (score >= 0.40) return { label: "Supporting", className: "bg-[#a78bfa]/20 text-[#a78bfa] border-[#a78bfa]/40" };
  return               { label: "Background",  className: "bg-white/10    text-white/50  border-white/20"  };
}

function friendlyError(err) {
  if (err.name === "TypeError" || err.message === "Failed to fetch") {
    return "Could not connect to the analysis server. Make sure the backend is running.";
  }
  return err.message || "An error occurred while analyzing the audio.";
}

function formatTime(t) {
  const m = Math.floor(t / 60);
  const s = Math.floor(t % 60).toString().padStart(2, "0");
  return `${m}:${s}`;
}

// ---------------------------------------------------------------------------
// Waveform player
// ---------------------------------------------------------------------------

function WaveformPlayer({ audioUrl }) {
  const containerRef = useRef(null);
  const wsRef = useRef(null);
  const [playing, setPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [ready, setReady] = useState(false);

  useEffect(() => {
    if (!audioUrl || !containerRef.current) return;

    if (wsRef.current) {
      wsRef.current.destroy();
      wsRef.current = null;
    }
    setPlaying(false);
    setCurrentTime(0);
    setDuration(0);
    setReady(false);

    const ws = WaveSurfer.create({
      container: containerRef.current,
      waveColor: "rgba(80, 100, 220, 0.6)",
      progressColor: "#00c8ff",
      cursorColor: "rgba(255,255,255,0.5)",
      height: 72,
      normalize: true,
      barWidth: 2,
      barGap: 1,
      barRadius: 2,
    });

    ws.load(audioUrl);

    ws.on("ready", (dur) => {
      setDuration(dur);
      setReady(true);
    });
    ws.on("timeupdate", (t) => setCurrentTime(t));
    ws.on("seeking", (t) => setCurrentTime(t));
    ws.on("play", () => setPlaying(true));
    ws.on("pause", () => setPlaying(false));
    ws.on("finish", () => {
      setPlaying(false);
      setCurrentTime(0);
    });

    wsRef.current = ws;

    return () => {
      ws.destroy();
      wsRef.current = null;
    };
  }, [audioUrl]);

  const togglePlay = () => wsRef.current?.playPause();

  return (
    <div className="w-full">
      <div ref={containerRef} className="w-full rounded-lg overflow-hidden" />
      <div className="flex items-center gap-3 mt-2">
        <button
          onClick={togglePlay}
          disabled={!ready}
          className="w-8 h-8 flex items-center justify-center rounded-full bg-[#006480] text-white text-xs transition-all hover:bg-[#00a6cc] disabled:opacity-40"
          aria-label={playing ? "Pause" : "Play"}
        >
          {playing ? <FaPause /> : <FaPlay />}
        </button>
        <span className="text-white/50 text-xs tabular-nums">
          {formatTime(currentTime)} / {formatTime(duration)}
        </span>
        {!ready && (
          <span className="text-white/30 text-xs">Loading waveform…</span>
        )}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Animated background
// ---------------------------------------------------------------------------

function AmbientBackground() {
  const canvasRef = useRef(null);
  const animationRef = useRef(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");

    const circleCount = 120;
    const circlePropCount = 8;
    const circlePropsLength = circleCount * circlePropCount;
    const baseSpeed = 0.5;
    const rangeSpeed = 2;
    const baseTTL = 150;
    const rangeTTL = 200;
    const baseRadius = 100;
    const rangeRadius = 200;
    const rangeHue = 50;
    const xOff = 0.0015;
    const yOff = 0.0015;
    const zOff = 0.0015;
    const TAU = Math.PI * 2;

    let circleProps = new Float32Array(circlePropsLength);
    const noise3D = createNoise3D();
    let baseHue = 250;

    const offCanvas = document.createElement("canvas");
    const offCtx = offCanvas.getContext("2d");

    const rand = (max) => Math.random() * max;

    const initCircle = (i) => {
      const x = rand(canvas.width);
      const y = rand(canvas.height);
      const n = noise3D(x * xOff, y * yOff, baseHue * zOff);
      const t = rand(TAU);
      const speed = baseSpeed + rand(rangeSpeed);
      const vx = speed * Math.cos(t);
      const vy = speed * Math.sin(t);
      const ttl = baseTTL + rand(rangeTTL);
      const radius = baseRadius + rand(rangeRadius);
      const hue = baseHue + n * rangeHue;
      circleProps.set([x, y, vx, vy, 0, ttl, radius, hue], i);
    };

    const reinitParticles = () => {
      for (let i = 0; i < circlePropsLength; i += circlePropCount) initCircle(i);
    };

    const setCanvasSize = () => {
      canvas.width = window.innerWidth;
      canvas.height = window.innerHeight;
      offCanvas.width = canvas.width / 2;
      offCanvas.height = canvas.height / 2;
    };

    setCanvasSize();
    reinitParticles();

    const handleResize = () => { setCanvasSize(); reinitParticles(); };
    window.addEventListener("resize", handleResize);

    const fadeInOut = (life, ttl) => {
      const half = ttl / 2;
      return life < half ? life / half : 1 - (life - half) / half;
    };

    const drawCircle = (ctxTarget, x, y, life, ttl, radius, hue) => {
      ctxTarget.fillStyle = `hsla(${hue},60%,30%,${fadeInOut(life, ttl)})`;
      ctxTarget.beginPath();
      ctxTarget.arc(x / 2, y / 2, radius / 2, 0, TAU);
      ctxTarget.fill();
    };

    const checkBounds = (x, y, radius) =>
      x < -radius || x > canvas.width + radius || y < -radius || y > canvas.height + radius;

    const updateCircles = () => {
      baseHue++;
      if (baseHue > 300) baseHue = 200;
      for (let i = 0; i < circlePropsLength; i += circlePropCount) {
        let x = circleProps[i], y = circleProps[i + 1];
        let vx = circleProps[i + 2], vy = circleProps[i + 3];
        let life = circleProps[i + 4], ttl = circleProps[i + 5];
        let radius = circleProps[i + 6], hue = circleProps[i + 7];
        drawCircle(offCtx, x, y, life, ttl, radius, hue);
        life++; x += vx; y += vy;
        if (checkBounds(x, y, radius) || life > ttl) initCircle(i);
        else circleProps.set([x, y, vx, vy, life, ttl, radius, hue], i);
      }
    };

    const draw = () => {
      ctx.fillStyle = "hsla(0,0%,5%,0.3)";
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      offCtx.clearRect(0, 0, offCanvas.width, offCanvas.height);
      updateCircles();
      ctx.filter = "blur(45px)";
      ctx.drawImage(offCanvas, 0, 0, canvas.width, canvas.height);
      ctx.filter = "none";
      ctx.globalCompositeOperation = "multiply";
      ctx.fillStyle = "rgba(0,0,0,0.4)";
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      ctx.globalCompositeOperation = "source-over";
      animationRef.current = requestAnimationFrame(draw);
    };

    const handleVisibilityChange = () => {
      if (document.hidden) cancelAnimationFrame(animationRef.current);
      else draw();
    };
    document.addEventListener("visibilitychange", handleVisibilityChange);
    draw();

    return () => {
      window.removeEventListener("resize", handleResize);
      document.removeEventListener("visibilitychange", handleVisibilityChange);
      cancelAnimationFrame(animationRef.current);
    };
  }, []);

  return (
    <canvas ref={canvasRef} className="fixed top-0 left-0 w-full h-full -z-10 pointer-events-none block" />
  );
}

// ---------------------------------------------------------------------------
// Main app
// ---------------------------------------------------------------------------

function App() {
  const [file, setFile] = useState(null);
  const [audioUrl, setAudioUrl] = useState("");
  const [scores, setScores] = useState({});
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState("");
  const [analyzed, setAnalyzed] = useState(false);
  const [isDragging, setIsDragging] = useState(false);

  const handleFile = useCallback((selectedFile) => {
    if (!selectedFile) return;
    setAudioUrl((prev) => { if (prev) URL.revokeObjectURL(prev); return URL.createObjectURL(selectedFile); });
    setFile(selectedFile);
    setScores({});
    setError("");
    setAnalyzed(false);
  }, []);

  const handleFileChange = (e) => handleFile(e.target.files[0]);
  const handleDrop = (e) => { e.preventDefault(); setIsDragging(false); handleFile(e.dataTransfer.files[0]); };
  const handleDragOver = (e) => { e.preventDefault(); setIsDragging(true); };
  const handleDragLeave = () => setIsDragging(false);

  const handleUpload = async () => {
    if (!file) return;
    setUploading(true);
    setError("");
    setAnalyzed(false);
    const formData = new FormData();
    formData.append("file", file);
    try {
      const response = await fetch(`${API_BASE}/api/predict`, { method: "POST", body: formData });
      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.error || `Server error (${response.status})`);
      }
      const data = await response.json();
      const normalizedScores = {};
      for (const key in (data.scores || {})) {
        normalizedScores[key] = Math.min(Math.max(parseFloat(data.scores[key]), 0), 1);
      }
      setScores(normalizedScores);
      setAnalyzed(true);
    } catch (err) {
      setError(friendlyError(err));
      setScores({});
    } finally {
      setUploading(false);
    }
  };

  const sortedScores = Object.entries(scores).sort(([, a], [, b]) => b - a);
  const detectedCount = sortedScores.length;

  return (
    <div className="relative max-w-3xl mx-auto my-8 px-4 text-center z-10">
      <AmbientBackground />

      <div className="flex flex-col items-center w-full gap-[18px] px-8 py-7 rounded-[18px] border border-[rgba(120,140,255,0.25)] bg-[rgba(12,14,22,0.55)] backdrop-blur-[16px] [-webkit-backdrop-filter:blur(16px)] shadow-[0_0_0_1px_rgba(120,140,255,0.08),0_20px_60px_rgba(0,0,0,0.6)]">

        <h1 className="text-white font-bold text-2xl m-0">&#9835; Instrument Detection</h1>

        {/* Introduction */}
        <div className="w-full text-left text-white/60 text-sm leading-relaxed space-y-2">
          <p>
            Upload any audio file and the model will identify which instruments are present.
            Results are sorted by how prominently each instrument appears in the recording.
          </p>
          <div className="grid grid-cols-3 gap-2 pt-1">
            {[
              { label: "Lead",        cls: "bg-[#00c8ff]/20 text-[#00c8ff] border-[#00c8ff]/40", desc: "Dominant — clearly audible throughout" },
              { label: "Supporting",  cls: "bg-[#a78bfa]/20 text-[#a78bfa] border-[#a78bfa]/40", desc: "Noticeable but not the main focus" },
              { label: "Background",  cls: "bg-white/10 text-white/50 border-white/20",           desc: "Subtle or intermittent" },
            ].map(({ label, cls, desc }) => (
              <div key={label} className="flex flex-col gap-2 p-3 rounded-xl bg-white/[0.03] border border-white/[0.06]">
                <span className={`self-start text-xs font-semibold px-3 py-1 rounded-full border ${cls}`}>{label}</span>
                <p className="text-white/40 text-xs leading-snug m-0">{desc}</p>
              </div>
            ))}
          </div>
        </div>

        {/* Drop zone */}
        <label
          className={[
            "flex justify-center items-center w-full min-h-[120px] rounded-[10px]",
            "border border-dashed cursor-pointer transition-[border-color,background] duration-300 bg-gradient-to-r",
            isDragging ? "border-[#00c8ff] from-[#004d70] to-[#6a1faf]" : "border-[#002c66] from-[#003652] to-[#4e1683]",
          ].join(" ")}
          onDrop={handleDrop}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
        >
          <div className="flex flex-col items-center text-center">
            <span className="text-xl text-white">{file ? file.name : "Drop an audio file here"}</span>
            {!file && <span className="text-sm text-[#999] mt-0.5">or click to browse files</span>}
          </div>
          <input type="file" accept=".wav,.mp3,.flac,.ogg,.m4a" onChange={handleFileChange} className="hidden" aria-label="Upload audio file" />
        </label>

        {/* Waveform player */}
        {audioUrl && <WaveformPlayer audioUrl={audioUrl} />}

        {/* Analyze button */}
        {file && (
          <button
            onClick={handleUpload}
            disabled={uploading}
            className="px-5 py-2.5 rounded-[10px] bg-[#006480] text-white font-bold text-base cursor-pointer inline-flex items-center justify-center gap-2 transition-all duration-300 hover:bg-[#00a6cc] hover:-translate-y-0.5 hover:shadow-[0_6px_16px_rgba(0,200,255,0.5)] disabled:opacity-60 disabled:cursor-not-allowed disabled:translate-y-0 disabled:shadow-[0_3px_8px_rgba(0,200,255,0.3)]"
          >
            {uploading ? (
              <><span className="w-3.5 h-3.5 rounded-full border-2 border-white/30 border-t-white animate-spin shrink-0" aria-hidden="true" />Analyzing...</>
            ) : "Analyze"}
          </button>
        )}

        {/* Error */}
        {error && (
          <div className="w-full text-[#ff6b6b] px-3 py-3 bg-[rgba(255,107,107,0.1)] rounded-lg border border-[rgba(255,107,107,0.3)] text-[0.95rem]">
            {error}
          </div>
        )}

        {/* Results */}
        {analyzed && (
          <div className="w-full text-left">

            {/* Heading */}
            <div className="flex items-center justify-between mb-3">
              <h2 className="text-white font-semibold text-lg m-0">
                {detectedCount > 0 ? `${detectedCount} Instrument${detectedCount !== 1 ? "s" : ""} Detected` : "No Instruments Detected"}
              </h2>
            </div>

            {/* Instrument list */}
            {sortedScores.length > 0 && (
              <ul className="list-none p-0 m-0">
                {sortedScores.map(([inst, score], index) => {
                  const role = getRole(score);
                  return (
                    <li
                      key={inst}
                      className="flex items-center mb-3.5 gap-4 opacity-0 translate-y-2.5 [animation:fadeInUp_0.4s_ease_forwards]"
                      style={{ animationDelay: `${index * 0.05}s` }}
                    >
                      <span className="text-[2rem] w-10 text-[#00c8ff] shrink-0">
                        {INSTRUMENT_ICONS[inst] || <FaMusic />}
                      </span>
                      <span className="flex-1 text-[1.1rem] capitalize text-white">
                        {DISPLAY_NAMES[inst] || inst}
                      </span>
                      <span className={`text-xs font-semibold px-3 py-1 rounded-full border ${role.className}`}>
                        {role.label}
                      </span>
                    </li>
                  );
                })}
              </ul>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

export default App;
