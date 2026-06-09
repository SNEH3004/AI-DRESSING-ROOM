const { useState, useEffect, useCallback, useRef } = React;

const API = "http://localhost:5000";

// ── Colour palette ──────────────────────────────────────────────────
const C = {
  bg:      "#0a0b0f",
  surface: "#12141a",
  panel:   "#181c24",
  border:  "#242836",
  accent:  "#00e5c8",
  gold:    "#f5c842",
  muted:   "#4a5068",
  text:    "#e8ecf4",
  sub:     "#8490b0",
};

// ── Pill badge ──────────────────────────────────────────────────────
function Badge({ label, value, accent = false }) {
  return (
    <div style={{
      display:       "flex",
      flexDirection: "column",
      alignItems:    "center",
      gap:           2,
    }}>
      <span style={{ fontSize: 10, color: C.sub, letterSpacing: "0.12em", textTransform: "uppercase" }}>
        {label}
      </span>
      <span style={{
        fontSize:   16,
        fontWeight: 700,
        color:      accent ? C.accent : C.text,
        fontFamily: "'DM Mono', monospace",
      }}>
        {value}
      </span>
    </div>
  );
}

// ── Confidence bar ──────────────────────────────────────────────────
function ConfidenceBar({ value }) {
  const pct  = Math.min(100, Math.max(0, value));
  const hue  = pct >= 75 ? "#00e5c8" : pct >= 50 ? "#f5a623" : "#e05252";

  return (
    <div style={{ width: "100%" }}>
      <div style={{
        display:        "flex",
        justifyContent: "space-between",
        marginBottom:   6,
        fontSize:       11,
        color:          C.sub,
        letterSpacing:  "0.1em",
        textTransform:  "uppercase",
      }}>
        <span>Gesture Confidence</span>
        <span style={{ color: hue, fontWeight: 700 }}>{pct.toFixed(0)}%</span>
      </div>
      <div style={{
        height:       8,
        borderRadius: 4,
        background:   C.border,
        overflow:     "hidden",
      }}>
        <div style={{
          height:     "100%",
          width:      `${pct}%`,
          background: `linear-gradient(90deg, ${hue}88, ${hue})`,
          borderRadius: 4,
          transition: "width 0.25s ease",
          boxShadow:  `0 0 10px ${hue}66`,
        }} />
      </div>
    </div>
  );
}

// ── Gesture indicator ───────────────────────────────────────────────
function GestureTag({ gesture }) {
  if (!gesture || gesture === "none") return null;

  const isRight = gesture === "swipe_right";
  const color   = isRight ? C.accent : C.gold;
  const label   = isRight ? "→ NEXT SHIRT" : "PREV SHIRT ←";

  return (
    <div style={{
      display:      "inline-flex",
      alignItems:   "center",
      gap:          8,
      padding:      "6px 16px",
      borderRadius: 999,
      border:       `1px solid ${color}55`,
      background:   `${color}18`,
      color:        color,
      fontSize:     13,
      fontWeight:   700,
      letterSpacing:"0.08em",
      animation:    "fadeSlide 0.25s ease",
    }}>
      {label}
    </div>
  );
}

// ── Shirt thumbnail dots ────────────────────────────────────────────
function CarouselDots({ current, total, onSelect }) {
  return (
    <div style={{ display: "flex", gap: 10, alignItems: "center", justifyContent: "center" }}>
      {Array.from({ length: total }).map((_, i) => (
        <button
          key={i}
          onClick={() => onSelect(i)}
          style={{
            width:        i === current ? 28 : 10,
            height:       10,
            borderRadius: 5,
            border:       "none",
            cursor:       "pointer",
            background:   i === current ? C.accent : C.border,
            boxShadow:    i === current ? `0 0 12px ${C.accent}88` : "none",
            transition:   "all 0.3s ease",
            padding:      0,
          }}
        />
      ))}
    </div>
  );
}

// ── Status strip ────────────────────────────────────────────────────
function StatusStrip({ fps, inCooldown }) {
  return (
    <div style={{
      display:    "flex",
      alignItems: "center",
      gap:        16,
      fontSize:   11,
      color:      C.muted,
    }}>
      <span style={{
        color:      fps >= 28 ? C.accent : "#f5a623",
        fontFamily: "'DM Mono', monospace",
        fontWeight: 600,
      }}>
        {fps} FPS
      </span>
      <span style={{
        width:       6,
        height:      6,
        borderRadius:"50%",
        background:  inCooldown ? "#f5a623" : C.accent,
        boxShadow:   `0 0 8px ${inCooldown ? "#f5a623" : C.accent}`,
        display:     "inline-block",
      }} />
      <span>{inCooldown ? "Cooldown…" : "Ready"}</span>
    </div>
  );
}

// ── Swipe hint ──────────────────────────────────────────────────────
function SwipeHint() {
  return (
    <div style={{
      display:      "flex",
      gap:          32,
      fontSize:     12,
      color:        C.muted,
      letterSpacing:"0.06em",
    }}>
      <span>← Swipe left to go back</span>
      <span>Swipe right to advance →</span>
    </div>
  );
}

// ── Main App ────────────────────────────────────────────────────────
export default function App() {
  const [status, setStatus] = useState({
    current_shirt: 0,
    total_shirts:  5,
    fps:           0,
    gesture:       "none",
    confidence:    0,
    in_cooldown:   false,
  });

  const pollRef  = useRef(null);
  const errCount = useRef(0);

  // Poll /status every 300 ms
  const poll = useCallback(async () => {
    try {
      const res = await fetch(`${API}/status`);
      if (!res.ok) throw new Error("non-200");
      const data = await res.json();
      setStatus(data);
      errCount.current = 0;
    } catch {
      errCount.current += 1;
    }
  }, []);

  useEffect(() => {
    poll();
    pollRef.current = setInterval(poll, 300);
    return () => clearInterval(pollRef.current);
  }, [poll]);

  const selectShirt = useCallback(async (idx) => {
    try {
      await fetch(`${API}/shirt/${idx}`, { method: "POST" });
      setStatus(s => ({ ...s, current_shirt: idx }));
    } catch {/* ignore */}
  }, []);

  return (
    <>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Syne:wght@600;700;800&family=Inter:wght@400;500;600&display=swap');
        *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
        html, body { background: ${C.bg}; color: ${C.text}; height: 100%; font-family: 'Inter', sans-serif; }
        @keyframes fadeSlide {
          from { opacity: 0; transform: translateY(6px); }
          to   { opacity: 1; transform: translateY(0); }
        }
        @keyframes pulse {
          0%, 100% { opacity: 1; }
          50%       { opacity: 0.5; }
        }
        ::-webkit-scrollbar { display: none; }
      `}</style>

      <div style={{
        minHeight:      "100vh",
        display:        "flex",
        flexDirection:  "column",
        alignItems:     "center",
        padding:        "24px 16px 40px",
        gap:            24,
        background:     `radial-gradient(ellipse 80% 60% at 50% 0%, #0e2029 0%, ${C.bg} 65%)`,
      }}>

        {/* ── Header ─────────────────────────────────────────────── */}
        <header style={{ textAlign: "center" }}>
          <div style={{
            display:      "inline-flex",
            alignItems:   "center",
            gap:          10,
            padding:      "5px 18px",
            borderRadius: 999,
            border:       `1px solid ${C.border}`,
            background:   C.surface,
            fontSize:     11,
            color:        C.sub,
            letterSpacing:"0.14em",
            marginBottom: 12,
            textTransform:"uppercase",
          }}>
            <span style={{ width: 6, height: 6, borderRadius: "50%", background: C.accent, boxShadow: `0 0 8px ${C.accent}` }} />
            AI Virtual Dressing Room
          </div>
          <h1 style={{
            fontFamily:  "'Syne', sans-serif",
            fontWeight:  800,
            fontSize:    "clamp(28px, 5vw, 42px)",
            letterSpacing:"-0.02em",
            lineHeight:  1.1,
            color:       C.text,
          }}>
            Try It On.{" "}
            <span style={{ color: C.accent }}>Instantly.</span>
          </h1>
          <p style={{ marginTop: 8, fontSize: 14, color: C.sub }}>
            Swipe your hand in front of the camera to change shirts
          </p>
        </header>

        {/* ── Main layout ────────────────────────────────────────── */}
        <div style={{
          display:             "grid",
          gridTemplateColumns: "1fr 320px",
          gap:                 20,
          width:               "100%",
          maxWidth:            1140,
        }}>

          {/* ── Video feed ─────────────────────────────────────── */}
          <div style={{
            position:     "relative",
            borderRadius: 20,
            overflow:     "hidden",
            border:       `1px solid ${C.border}`,
            background:   C.surface,
            aspectRatio:  "16/9",
          }}>
            <img
              src={`${API}/video_feed`}
              alt="Dressing room camera feed"
              style={{ width: "100%", height: "100%", objectFit: "cover", display: "block" }}
            />
            {/* Vignette */}
            <div style={{
              position:   "absolute",
              inset:      0,
              background: "radial-gradient(ellipse at center, transparent 55%, #0a0b0f88 100%)",
              pointerEvents:"none",
            }} />
          </div>

          {/* ── Side panel ─────────────────────────────────────── */}
          <aside style={{
            display:       "flex",
            flexDirection: "column",
            gap:           16,
          }}>

            {/* Shirt indicator */}
            <div style={{
              background:   C.panel,
              border:       `1px solid ${C.border}`,
              borderRadius: 16,
              padding:      "22px 20px",
              display:      "flex",
              flexDirection:"column",
              gap:          18,
              alignItems:   "center",
            }}>
              <div style={{
                fontFamily:  "'Syne', sans-serif",
                fontWeight:  700,
                fontSize:    36,
                color:       C.accent,
                lineHeight:  1,
              }}>
                {status.current_shirt + 1}
                <span style={{ fontSize: 18, color: C.muted, fontWeight: 600 }}>
                  {" "}/ {status.total_shirts}
                </span>
              </div>
              <CarouselDots
                current={status.current_shirt}
                total={status.total_shirts}
                onSelect={selectShirt}
              />
              <div style={{ fontSize: 12, color: C.muted }}>
                Tap a dot to jump to that shirt
              </div>
            </div>

            {/* Gesture panel */}
            <div style={{
              background:   C.panel,
              border:       `1px solid ${C.border}`,
              borderRadius: 16,
              padding:      "18px 20px",
              display:      "flex",
              flexDirection:"column",
              gap:          14,
            }}>
              <div style={{
                fontSize:     11,
                color:        C.sub,
                letterSpacing:"0.12em",
                textTransform:"uppercase",
              }}>
                Gesture Status
              </div>

              <GestureTag gesture={status.gesture} />
              <ConfidenceBar value={status.confidence} />
            </div>

            {/* Stats row */}
            <div style={{
              background:   C.panel,
              border:       `1px solid ${C.border}`,
              borderRadius: 16,
              padding:      "16px 20px",
              display:      "grid",
              gridTemplateColumns: "1fr 1fr",
              gap:          12,
            }}>
              <Badge label="FPS" value={status.fps} accent />
              <Badge label="Mode" value={status.in_cooldown ? "Wait" : "Active"} />
            </div>

            {/* Status strip */}
            <StatusStrip fps={status.fps} inCooldown={status.in_cooldown} />

            {/* Divider */}
            <div style={{ height: 1, background: C.border }} />

            {/* Gesture guide */}
            <div style={{
              background:   C.panel,
              border:       `1px solid ${C.border}`,
              borderRadius: 16,
              padding:      "16px 20px",
              display:      "flex",
              flexDirection:"column",
              gap:          10,
            }}>
              <div style={{ fontSize: 11, color: C.sub, letterSpacing: "0.12em", textTransform: "uppercase" }}>
                How to Swipe
              </div>
              {[
                ["→", "Swipe Right", "Next shirt"],
                ["←", "Swipe Left",  "Previous shirt"],
              ].map(([icon, name, desc]) => (
                <div key={name} style={{ display: "flex", gap: 12, alignItems: "center" }}>
                  <span style={{ fontSize: 20, color: C.accent, width: 24, textAlign: "center" }}>{icon}</span>
                  <div>
                    <div style={{ fontSize: 13, fontWeight: 600, color: C.text }}>{name}</div>
                    <div style={{ fontSize: 11, color: C.muted }}>{desc}</div>
                  </div>
                </div>
              ))}
              <div style={{ fontSize: 11, color: C.muted, marginTop: 4, lineHeight: 1.6 }}>
                Move index finger &gt;120 px horizontally within 0.3–0.8 s.
                A 1-second cooldown follows each detected swipe.
              </div>
            </div>
          </aside>
        </div>

        {/* ── Footer hint ─────────────────────────────────────────── */}
        <SwipeHint />
      </div>
    </>
  );
}
