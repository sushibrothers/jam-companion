import os
import json
import base64
import numpy as np
import librosa
import streamlit as st

st.set_page_config(page_title="Jam Companion Pro", page_icon="🎸", layout="wide")

PITCH_CLASSES = tuple(('C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'))
GUITAR_TUNING = tuple(('E', 'B', 'G', 'D', 'A', 'E'))

MAJOR_PROFILE = np.array((6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88))
MINOR_PROFILE = np.array((6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17))

CHORD_DEFINITIONS = {
    "": ("Mayor", (0, 4, 7)),
    "m": ("menor", (0, 3, 7)),
    "7": ("7ma Dominante", (0, 4, 7, 10)),
    "maj7": ("Mayor 7ma", (0, 4, 7, 11)),
    "m7": ("menor 7ma", (0, 3, 7, 10)),
}

def build_chord_templates():
    templates = {}
    for i, root in enumerate(PITCH_CLASSES):
        maj = np.zeros(12)
        maj[(i + 0) % 12] = 1.0; maj[(i + 4) % 12] = 1.0; maj[(i + 7) % 12] = 1.0
        templates[f"{root}"] = maj / np.linalg.norm(maj)

        minor = np.zeros(12)
        minor[(i + 0) % 12] = 1.0; minor[(i + 3) % 12] = 1.0; minor[(i + 7) % 12] = 1.0
        templates[f"{root}m"] = minor / np.linalg.norm(minor)

        dom7 = np.zeros(12)
        dom7[(i + 0) % 12] = 1.0; dom7[(i + 4) % 12] = 1.0; dom7[(i + 7) % 12] = 1.0; dom7[(i + 10) % 12] = 0.8
        templates[f"{root}7"] = dom7 / np.linalg.norm(dom7)
    return templates

def pearson_correlation(x, y):
    x_diff = x - np.mean(x)
    y_diff = y - np.mean(y)
    denom = (np.sqrt(np.sum(x_diff ** 2)) * np.sqrt(np.sum(y_diff ** 2))) + 1e-9
    return float(np.sum(x_diff * y_diff) / denom)

def detect_key(chroma_mean):
    correlations = {}
    for i, root in enumerate(PITCH_CLASSES):
        maj_prof = np.roll(MAJOR_PROFILE, i)
        min_prof = np.roll(MINOR_PROFILE, i)
        correlations[f"{root} Mayor"] = pearson_correlation(chroma_mean, maj_prof)
        correlations[f"{root} menor"] = pearson_correlation(chroma_mean, min_prof)
    best_key, score = max(correlations.items(), key=lambda item: item)
    return best_key, float(score)

def parse_chord(chord_str):
    if not chord_str or chord_str == "N":
        return None, None
    if len(chord_str) > 1 and chord_str.startswith(tuple(c for c in PITCH_CLASSES if "#" in c)):
        return chord_str[:2], chord_str[2:]
    return chord_str[:1], chord_str[1:]

def get_arpeggio_details(chord_str):
    root, quality = parse_chord(chord_str)
    if not root or root not in PITCH_CLASSES:
        return None, tuple()
    root_idx = PITCH_CLASSES.index(root)
    name_qual, intervals = CHORD_DEFINITIONS.get(quality, CHORD_DEFINITIONS[""])
    notes = tuple(PITCH_CLASSES[(root_idx + interval) % 12] for interval in intervals)
    return root, notes

def get_harmonic_degree(chord_str, key_root, is_major):
    root, _ = parse_chord(chord_str)
    if not root or key_root not in PITCH_CLASSES or root not in PITCH_CLASSES:
        return "-"
    key_idx = PITCH_CLASSES.index(key_root)
    root_idx = PITCH_CLASSES.index(root)
    semitones = (root_idx - key_idx) % 12
    if is_major:
        mapping = {0: "I", 2: "ii", 4: "iii", 5: "IV", 7: "V", 9: "vi", 11: "vii°"}
    else:
        mapping = {0: "i", 2: "ii°", 3: "III", 5: "iv", 7: "v", 8: "VI", 10: "VII"}
    return mapping.get(semitones, f"b{semitones}")

def match_chord(chroma_vector, templates, min_energy=0.01):
    norm = np.linalg.norm(chroma_vector)
    if norm < min_energy:
        return "N"
    normalized_vec = chroma_vector / norm
    best_chord = "N"
    best_sim = -1.0
    for chord_name, template_vec in templates.items():
        sim = np.dot(normalized_vec, template_vec)
        if sim > best_sim:
            best_sim = sim
            best_chord = chord_name
    return best_chord

def render_interactive_synced_player(audio_base64, audio_type, segments, scale_notes):
    segments_json = json.dumps(segments)
    scale_notes_json = json.dumps(scale_notes)
    tuning_json = json.dumps(list(GUITAR_TUNING))
    pitch_classes_json = json.dumps(list(PITCH_CLASSES))
    
    html_code = f"""
    <!DOCTYPE html>
    <html>
    <head>
      <meta charset="utf-8">
      <style>
        body {{
          margin: 0;
          padding: 8px;
          background: #0f172a;
          color: #f8fafc;
          font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        }}
        .card {{
          background: #1e293b;
          border: 1px solid #334155;
          border-radius: 12px;
          padding: 16px;
          margin-bottom: 12px;
        }}
        .player-container {{
          display: flex;
          flex-direction: column;
          gap: 12px;
        }}
        audio {{
          width: 100%;
          border-radius: 8px;
          outline: none;
        }}
        .status-box {{
          display: flex;
          justify-content: space-around;
          align-items: center;
          background: #0f172a;
          border: 2px solid #6366f1;
          border-radius: 10px;
          padding: 12px;
          text-align: center;
        }}
        .chord-title {{
          font-size: 34px;
          font-weight: 800;
          color: #38bdf8;
        }}
        .badge {{
          display: inline-block;
          padding: 4px 10px;
          margin: 2px;
          border-radius: 6px;
          font-weight: bold;
          font-size: 13px;
        }}
        .badge-root {{ background: #f59e0b; color: #000; }}
        .badge-arp {{ background: #10b981; color: #fff; }}
        .timeline-scroll {{
          display: flex;
          gap: 8px;
          overflow-x: auto;
          padding: 6px 0;
        }}
        .timeline-item {{
          flex: 0 0 auto;
          background: #334155;
          border: 1px solid #475569;
          border-radius: 8px;
          padding: 8px 12px;
          text-align: center;
          cursor: pointer;
          transition: all 0.2s;
        }}
        .timeline-item.active {{
          background: #4f46e5;
          border-color: #38bdf8;
          transform: scale(1.05);
          box-shadow: 0 0 10px rgba(56, 189, 248, 0.5);
        }}
      </style>
    </head>
    <body>
      <div class="card player-container">
        <audio id="audioPlayer" controls src="data:{audio_type};base64,{audio_base64}"></audio>
        
        <div class="status-box">
          <div>
            <div style="font-size: 11px; text-transform: uppercase; color: #94a3b8;">Acorde en Vivo</div>
            <div id="liveChord" class="chord-title">-</div>
            <div id="liveDegree" style="font-size: 13px; color: #cbd5e1;">Grado: -</div>
          </div>
          <div>
            <div style="font-size: 11px; text-transform: uppercase; color: #94a3b8; margin-bottom: 4px;">Notas para Arpegiar</div>
            <div id="liveArpeggio"></div>
          </div>
        </div>

        <div>
          <div style="font-size: 12px; font-weight: bold; color: #94a3b8; margin-bottom: 6px;">Línea temporal de acordes (Haz clic para saltar):</div>
          <div class="timeline-scroll" id="timelineList"></div>
        </div>
      </div>

      <div class="card" style="padding: 12px;">
        <div id="fretboardContainer"></div>
      </div>

      <script>
        const segments = {segments_json};
        const scaleNotes = {scale_notes_json};
        const tuning = {tuning_json};
        const pitchClasses = {pitch_classes_json};

        const audio = document.getElementById('audioPlayer');
        const liveChord = document.getElementById('liveChord');
        const liveDegree = document.getElementById('liveDegree');
        const liveArpeggio = document.getElementById('liveArpeggio');
        const timelineList = document.getElementById('timelineList');
        const fretboardContainer = document.getElementById('fretboardContainer');

        segments.forEach((seg, idx) => {{
          const item = document.createElement('div');
          item.className = 'timeline-item';
          item.id = 'seg-' + idx;
          item.innerHTML = `<div style="font-weight:bold; font-size:15px;">${{seg.chord}}</div><div style="font-size:10px; color:#cbd5e1;">${{seg.start}}s</div>`;
          item.onclick = () => {{
            audio.currentTime = seg.start;
            audio.play();
          }};
          timelineList.appendChild(item);
        }});

        function drawFretboard(rootNote, arpeggioNotes) {{
          const numFrets = 12;
          const width = 850, height = 220;
          const ml = 50, mr = 30, mt = 35, mb = 30;
          const fretWidth = (width - ml - mr) / numFrets;
          const stringHeight = (height - mt - mb) / 5;

          let svg = `<svg viewBox="0 0 ${{width}} ${{height}}" width="100%" height="auto" style="background:#0f172a; border-radius: 12px; font-family: system-ui, sans-serif;">`;
          svg += `<text x="${{ml}}" y="22" fill="#e2e8f0" font-size="13px" font-weight="bold">MÁSTIL DINÁMICO (Trastes 0 a ${{numFrets}})</text>`;
          svg += `<circle cx="${{width - 270}}" cy="18" r="6" fill="#f59e0b"/><text x="${{width - 258}}" y="22" fill="#94a3b8" font-size="11px">Fundamental</text>`;
          svg += `<circle cx="${{width - 170}}" cy="18" r="6" fill="#10b981"/><text x="${{width - 158}}" y="22" fill="#94a3b8" font-size="11px">Arpegio</text>`;
          svg += `<circle cx="${{width - 80}}" cy="18" r="5" fill="#334155"/><text x="${{width - 70}}" y="22" fill="#94a3b8" font-size="11px">Escala</text>`;

          svg += `<rect x="${{ml - 6}}" y="${{mt}}" width="6" height="${{stringHeight * 5}}" fill="#cbd5e1" rx="2" />`;

          for (let f = 1; f <= numFrets; f++) {{
            let x = ml + f * fretWidth;
            svg += `<line x1="${{x}}" y1="${{mt}}" x2="${{x}}" y2="${{mt + stringHeight * 5}}" stroke="#475569" stroke-width="2"/>`;
            svg += `<text x="${{x - fretWidth / 2}}" y="${{height - 10}}" fill="#64748b" font-size="11px" font-weight="bold" text-anchor="middle">${{f}}</text>`;
          }}

         .forEach(f => {{
            let cx = ml + (f - 0.5) * fretWidth;
            svg += `<circle cx="${{cx}}" cy="${{mt + 2.5 * stringHeight}}" r="5" fill="#334155" opacity="0.7"/>`;
          }});
          let cx12 = ml + 11.5 * fretWidth;
          svg += `<circle cx="${{cx12}}" cy="${{mt + 1.25 * stringHeight}}" r="4" fill="#334155" opacity="0.7"/>`;
          svg += `<circle cx="${{cx12}}" cy="${{mt + 3.75 * stringHeight}}" r="4" fill="#334155" opacity="0.7"/>`;

          tuning.forEach((openNote, sIdx) => {{
            let y = mt + sIdx * stringHeight;
            let th = 1.0 + (sIdx * 0.45);
            svg += `<line x1="${{ml}}" y1="${{y}}" x2="${{width - mr}}" y2="${{y}}" stroke="#94a3b8" stroke-width="${{th}}"/>`;
            svg += `<text x="${{ml - 18}}" y="${{y + 4}}" fill="#f8fafc" font-size="12px" font-weight="bold" text-anchor="middle">${{openNote}}</text>`;
          }});

          tuning.forEach((openNote, sIdx) => {{
            let openIdx = pitchClasses.indexOf(openNote);
            let y = mt + sIdx * stringHeight;
            for (let f = 0; f <= numFrets; f++) {{
              let note = pitchClasses[(openIdx + f) % 12];
              let cx = (f === 0) ? (ml - 18) : (ml + (f - 0.5) * fretWidth);

              if (rootNote && note === rootNote) {{
                svg += `<circle cx="${{cx}}" cy="${{y}}" r="11" fill="#f59e0b" stroke="#ffffff" stroke-width="1.5"/>`;
                svg += `<text x="${{cx}}" y="${{y + 4}}" fill="#000000" font-size="10px" font-weight="bold" text-anchor="middle">${{note}}</text>`;
              }} else if (arpeggioNotes && arpeggioNotes.includes(note)) {{
                svg += `<circle cx="${{cx}}" cy="${{y}}" r="10" fill="#10b981" stroke="#ffffff" stroke-width="1.2"/>`;
                svg += `<text x="${{cx}}" y="${{y + 4}}" fill="#ffffff" font-size="10px" font-weight="bold" text-anchor="middle">${{note}}</text>`;
              }} else if (scaleNotes && scaleNotes.includes(note)) {{
                svg += `<circle cx="${{cx}}" cy="${{y}}" r="8" fill="#1e293b" stroke="#475569" stroke-width="1"/>`;
                svg += `<text x="${{cx}}" y="${{y + 3.5}}" fill="#94a3b8" font-size="9px" text-anchor="middle">${{note}}</text>`;
              }}
            }}
          }});

          svg += `</svg>`;
          fretboardContainer.innerHTML = svg;
        }}

        let lastActiveIdx = -1;

        function onTick() {{
          const t = audio.currentTime;
          let activeIdx = -1;
          for (let i = 0; i < segments.length; i++) {{
            if (t >= segments[i].start && t < segments[i].end) {{
              activeIdx = i;
              break;
            }}
          }}

          if (activeIdx !== -1 && activeIdx !== lastActiveIdx) {{
            lastActiveIdx = activeIdx;
            const seg = segments[activeIdx];

            liveChord.innerText = seg.chord;
            liveDegree.innerText = 'Grado: ' + seg.degree;

            let badgesHtml = '';
            seg.arpeggio.forEach(n => {{
              badgesHtml += `<span class="badge ${{n === seg.root ? 'badge-root' : 'badge-arp'}}">${{n}}</span>`;
            }});
            liveArpeggio.innerHTML = badgesHtml;

            document.querySelectorAll('.timeline-item').forEach(el => el.classList.remove('active'));
            const activeEl = document.getElementById('seg-' + activeIdx);
            if (activeEl) {{
              activeEl.classList.add('active');
              activeEl.scrollIntoView({{ behavior: 'smooth', block: 'nearest', inline: 'center' }});
            }}

            drawFretboard(seg.root, seg.arpeggio);
          }}
          requestAnimationFrame(onTick);
        }}

        if (segments.length > 0) {{
          drawFretboard(segments[0].root, segments[0].arpeggio);
          liveChord.innerText = segments[0].chord;
          liveDegree.innerText = 'Grado: ' + segments[0].degree;
          let initialBadges = '';
          segments[0].arpeggio.forEach(n => {{
            initialBadges += `<span class="badge ${{n === segments[0].root ? 'badge-root' : 'badge-arp'}}">${{n}}</span>`;
          }});
          liveArpeggio.innerHTML = initialBadges;
        }} else {{
          drawFretboard(null, []);
        }}

        audio.addEventListener('play', () => requestAnimationFrame(onTick));
        audio.addEventListener('timeupdate', onTick);
      </script>
    </body>
    </html>
    """
    return html_code

st.title("🎸 Jam Companion: Escalas, Acordes y Mástil Sincronizado")
st.write("Sube tu canción para analizarla y ver cómo el mástil cambia dinámicamente al reproducir la pista.")

uploaded_file = st.file_uploader("Elige un archivo de audio (MP3 o WAV)", type=["mp3", "wav"])

if uploaded_file is not None:
    if st.button("🔍 Analizar Canción"):
        with st.spinner("Analizando armónicos y acordes..."):
            temp_path = f"temp_{uploaded_file.name}"
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            y, sr = librosa.load(temp_path, sr=22050, mono=True)
            duration = float(librosa.get_duration(y=y, sr=sr))
            y_harmonic, _ = librosa.effects.hpss(y)

            tempo, beat_frames = librosa.beat.beat_track(y=y_harmonic, sr=sr)
            tempo_val = float(np.asarray(tempo).flat[0])

            chroma = librosa.feature.chroma_cqt(y=y_harmonic, sr=sr)
            chroma_mean = np.mean(chroma, axis=1)
            key_name, key_score = detect_key(chroma_mean)

            tokens = key_name.split()
            key_root = tokens[0]
            is_major = "Mayor" in key_name
            root_idx = PITCH_CLASSES.index(key_root)

            if is_major:
                scale_steps = (0, 2, 4, 5, 7, 9, 11)
                penta_steps = (0, 2, 4, 7, 9)
                scale_notes = [PITCH_CLASSES[(root_idx + s) % 12] for s in scale_steps]
                penta_notes = [PITCH_CLASSES[(root_idx + s) % 12] for s in penta_steps]
                rel_key = f"{PITCH_CLASSES[(root_idx + 9) % 12]} menor"
            else:
                scale_steps = (0, 2, 3, 5, 7, 8, 10)
                penta_steps = (0, 3, 5, 7, 10)
                scale_notes = [PITCH_CLASSES[(root_idx + s) % 12] for s in scale_steps]
                penta_notes = [PITCH_CLASSES[(root_idx + s) % 12] for s in penta_steps]
                rel_key = f"{PITCH_CLASSES[(root_idx + 3) % 12]} Mayor"

            beat_chroma = librosa.util.sync(chroma, beat_frames, aggregate=np.median)
            beat_times = librosa.frames_to_time(beat_frames, sr=sr)
            times = np.concatenate([[0.0], beat_times, [duration]])

            templates = build_chord_templates()
            num_beats = beat_chroma.shape[-1]
            raw_chords = [match_chord(beat_chroma[:, b], templates) for b in range(num_beats)]

            segments = []
            if raw_chords:
                curr_chord = raw_chords[0]
                start_t = times[0]
                for i in range(1, len(raw_chords)):
                    if raw_chords[i] != curr_chord:
                        end_t = times[i]
                        if curr_chord != "N":
                            root, arp_notes = get_arpeggio_details(curr_chord)
                            degree = get_harmonic_degree(curr_chord, key_root, is_major)
                            segments.append({
                                "start": round(float(start_t), 2),
                                "end": round(float(end_t), 2),
                                "chord": curr_chord,
                                "root": root,
                                "degree": degree,
                                "arpeggio": list(arp_notes)
                            })
                        curr_chord = raw_chords[i]
                        start_t = end_t
                if curr_chord != "N":
                    root, arp_notes = get_arpeggio_details(curr_chord)
                    degree = get_harmonic_degree(curr_chord, key_root, is_major)
                    segments.append({
                        "start": round(float(start_t), 2),
                        "end": round(float(duration), 2),
                        "chord": curr_chord,
                        "root": root,
                        "degree": degree,
                        "arpeggio": list(arp_notes)
                    })

            if os.path.exists(temp_path):
                os.remove(temp_path)

            audio_b64 = base64.b64encode(uploaded_file.getvalue()).decode()
            audio_mime = uploaded_file.type or "audio/mp3"

            st.session_state['analysis_done'] = True
            st.session_state['key_name'] = key_name
            st.session_state['key_score'] = key_score
            st.session_state['tempo_val'] = tempo_val
            st.session_state['scale_notes'] = scale_notes
            st.session_state['penta_notes'] = penta_notes
            st.session_state['rel_key'] = rel_key
            st.session_state['segments'] = segments
            st.session_state['audio_b64'] = audio_b64
            st.session_state['audio_mime'] = audio_mime

if st.session_state.get('analysis_done', False):
    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🎵 Tonalidad y Tempo")
        st.write(f"**Tonalidad:** `{st.session_state['key_name']}` (Confianza: {st.session_state['key_score']:.2f})")
        st.write(f"**Tempo:** `{st.session_state['tempo_val']:.1f} BPM`")
    with col2:
        st.subheader("🎼 Escalas para Improvisar")
        st.write(f"**Escala Principal:** `{', '.join(st.session_state['scale_notes'])}`")
        st.write(f"**Pentatónica:** `{', '.join(st.session_state['penta_notes'])}`")
        st.write(f"**Tonalidad Relativa:** `{st.session_state['rel_key']}`")

    st.subheader("▶️ Reproductor y Mástil Sincronizado en Tiempo Real")
    player_html = render_interactive_synced_player(
        st.session_state['audio_b64'],
        st.session_state['audio_mime'],
        st.session_state['segments'],
        st.session_state['scale_notes']
    )
    st.components.v1.html(player_html, height=520)
