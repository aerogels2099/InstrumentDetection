import React, { useState, useRef, useEffect } from "react";
import "./App.css";
import { FaGuitar, FaDrum, FaMicrophone, FaMusic } from "react-icons/fa";
import { GiFlute, GiTrumpet, GiSaxophone, GiViolin } from "react-icons/gi";
import { LuPiano } from "react-icons/lu";
import { createNoise3D } from "simplex-noise";

const DISPLAY_NAMES = {
  bass: "Bass Guitar",
  cello: "Cello",
  clarinet: "Clarinet",
  drum: "Drums",
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
  drum: <FaDrum />,
  gac: <FaGuitar />,
  gel: <FaGuitar />,
  organ: <LuPiano />,
  piano: <LuPiano />,
  saxophone: <GiSaxophone />,
  trumpet: <GiTrumpet />,
  violin: <GiViolin />,
  voice: <FaMicrophone />,
};

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
    const simplex = { noise3D };
    let baseHue = 250;

    const offCanvas = document.createElement("canvas");
    const offCtx = offCanvas.getContext("2d");

    const resizeCanvas = () => {
      canvas.width = window.innerWidth;
      canvas.height = window.innerHeight;
      offCanvas.width = canvas.width / 2;
      offCanvas.height = canvas.height / 2;
    };
    resizeCanvas();
    window.addEventListener("resize", resizeCanvas);

    const rand = (max) => Math.random() * max;

    const initCircle = (i) => {
      const x = rand(canvas.width);
      const y = rand(canvas.height);
      const n = simplex.noise3D(x * xOff, y * yOff, baseHue * zOff);
      const t = rand(TAU);
      const speed = baseSpeed + rand(rangeSpeed);
      const vx = speed * Math.cos(t);
      const vy = speed * Math.sin(t);
      const life = 0;
      const ttl = baseTTL + rand(rangeTTL);
      const radius = baseRadius + rand(rangeRadius);
      const hue = baseHue + n * rangeHue;
      circleProps.set([x, y, vx, vy, life, ttl, radius, hue], i);
    };

    for (let i = 0; i < circlePropsLength; i += circlePropCount) {
      initCircle(i);
    }

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
        const idx = i;
        let x = circleProps[idx];
        let y = circleProps[idx + 1];
        let vx = circleProps[idx + 2];
        let vy = circleProps[idx + 3];
        let life = circleProps[idx + 4];
        let ttl = circleProps[idx + 5];
        let radius = circleProps[idx + 6];
        let hue = circleProps[idx + 7];

        drawCircle(offCtx, x, y, life, ttl, radius, hue);

        life++;
        x += vx;
        y += vy;

        if (checkBounds(x, y, radius) || life > ttl) {
          initCircle(i);
        } else {
          circleProps.set([x, y, vx, vy, life, ttl, radius, hue], idx);
        }
      }
    };

    const draw = () => {
      ctx.fillStyle = "hsla(0,0%,5%,0.3)"; // lower alpha
      ctx.fillRect(0, 0, canvas.width, canvas.height);

      offCtx.clearRect(0, 0, offCanvas.width, offCanvas.height);

      updateCircles();

      ctx.filter = "blur(45px)";
      ctx.drawImage(offCanvas, 0, 0, canvas.width, canvas.height);
      ctx.filter = "none";

      ctx.globalCompositeOperation = "multiply";
      ctx.fillStyle = "rgba(0, 0, 0, 0.4)";
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      ctx.globalCompositeOperation = "source-over";

      animationRef.current = requestAnimationFrame(draw);
    };

    draw();

    return () => {
      window.removeEventListener("resize", resizeCanvas);
      cancelAnimationFrame(animationRef.current);
    };
  }, []);

  return <canvas ref={canvasRef} className="ambient-canvas" style={{ display: "block" }} />;
}

function App() {
  const [file, setFile] = useState(null);
  const [audioUrl, setAudioUrl] = useState("");
  const [instruments, setInstruments] = useState([]);
  const [scores, setScores] = useState({});
  const [uploading, setUploading] = useState(false);

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    if (!selectedFile) return;
    setFile(selectedFile);
    setAudioUrl(URL.createObjectURL(selectedFile));
    setInstruments([]);
    setScores({});
  };

  const handleUpload = async () => {
    if (!file) return;
    setUploading(true);
    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await fetch("http://127.0.0.1:8000/predict", {
        method: "POST",
        body: formData,
      });
      const data = await response.json();

      const detected = Array.isArray(data.detected_instruments)
        ? data.detected_instruments
        : [];
      const probScores = data.scores || {};
      const normalizedScores = {};
      for (const key in probScores) {
        normalizedScores[key] = Math.min(Math.max(parseFloat(probScores[key]), 0), 1);
      }

      setInstruments(detected);
      setScores(normalizedScores);
    } catch (err) {
      console.error(err);
      alert("Error uploading file!");
      setInstruments([]);
      setScores({});
    } finally {
      setUploading(false);
    }
  };

  return (
    <div className="App">
      <AmbientBackground />
      <div className="content-overlay">
        <h1>♫ Instrument Detection</h1>
        <label className="file-dropzone">
          <div className="file-dropzone-text">
            <span className="main-text">{file ? file.name : "Drop an audio file here"}</span>
            {!file && <span className="sub-text">or click to browse files</span>}
          </div>
          <input type="file" accept=".wav,.mp3" onChange={handleFileChange} />
        </label>
        {audioUrl && (
          <audio
            src={audioUrl}
            controls
            style={{
              width: "100%",
              marginTop: "16px",
              borderRadius: "12px"
            }}
          />
        )}
        {file && (
          <button onClick={handleUpload} disabled={uploading}>
            {uploading ? "Analyzing..." : "Analyze"}
          </button>
        )}
        {instruments.length > 0 && (
          <div className="visualization">
            <h2>Detected Instruments:</h2>
            <ul>
              {instruments.map((inst) => (
                <li key={inst} className="instrument-item">
                  <span className="icon">{INSTRUMENT_ICONS[inst] || <FaMusic />}</span>
                  <span className="instrument-name">{DISPLAY_NAMES[inst] || inst}</span>
                  <div
                    className="score-bar"
                    style={{ width: `${scores[inst] * 100}%`, backgroundColor: "#56c6ccff" }}
                  ></div>
                </li>
              ))}
            </ul>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;