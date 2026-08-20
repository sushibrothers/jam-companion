import os
import numpy as np
import librosa
import streamlit as st

st.set_page_config(page_title="Jam Companion Pro", page_icon="🎸", layout="wide")

PITCH_CLASSES = tuple(('C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'))
GUITAR_TUNING = tuple(('E', 'B', 'G', 'D', 'A', 'E'))

PROFILE_MAJ = np.array((1.0, 0.05, 0.4, 0.05, 0.75, 0.6, 0.05, 0.85, 0.05, 0.55, 0.1, 0.45))
PROFILE_MIN = np.array((1.0, 0.05, 0.4, 0.75, 0.05, 0.6, 0.05, 0.85, 0.6, 0.05, 0.5, 0.3))

CHORD_DEFINITIONS = {
    "": ("Mayor", (0, 4, 7)),
    "m": ("menor", (0, 3, 7)),
    "7": ("7ma Dominante", (0, 4, 7, 10)),
    "maj7": ("Mayor 7ma", (0, 4, 7, 11)),
    "m7": ("menor 7ma", (0, 3, 7, 10)),
}

def build_smart_chord_templates():
    templates = {}
    for i, root in enumerate(PITCH_CLASSES):
        maj = np.full(12, -0.3)
        maj[(i + 0) % 12] = 1.5; maj[(i + 4) % 12] = 1.0; maj[(i + 7) % 12] = 1.0
        templates[f"{root}"] = maj / np.linalg.norm(maj)

        minor = np.full(12, -0.3)
        minor[(i + 0) % 12] = 1.5; minor[(i + 3) % 12] = 1.0; minor[(i + 7) % 12] = 1.0
        templates[f"{root}m"] = minor / np.linalg.norm(minor)

        dom7 = np.full(12, -0.4)
        dom7[(i + 0) % 12] = 1.5; dom7[(i + 4) % 12] = 1.0; dom7[(i + 7) % 12] = 1.0; dom7[(i + 10) % 12] = 0.8
        templates[f"{root}7"] = dom7 / np.linalg.norm(dom7)
    return templates

def standardize_vec(v):
    return (v - np.mean(v)) / (np.std(v) + 1e-9)

def detect_key_accurate(chroma_matrix):
    chroma_mean = np.mean(chroma_matrix, axis=1)
    c_std = standardize_vec(chroma_mean)
    correlations = {}
    for i, root in enumerate(PITCH_CLASSES):
        maj_prof = standardize_vec(np.roll(PROFILE_MAJ, i))
        min_prof = standardize_vec(np.roll(PROFILE_MIN, i))
        correlations[f"{root} Mayor"] = float(np.dot(c_std, maj_prof) / 12.0)
        correlations[f"{root} menor"] = float(np.dot(c_std, min_prof) / 12.0)
    best_key, score = max(correlations.items(), key=lambda x: x[-1])
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

def get_pentatonic_box_range(key_root, is_major, box_number):
    root_idx = PITCH_CLASSES.index(key_root)
    min_root_idx = (root_idx + 9) % 12 if is_major else root_idx
    base_fret = (min_root_idx - 4) % 12
    offsets = {1: (0, 3), 2: (2, 5), 3: (5, 8), 4: (7, 10), 5: (9, 12)}
    start_off, end_off = offsets.get(box_number, (0, 15))
    start_f = (base_fret + start_off) % 12
    return start_f, start_f + (end_off - start_off)

def generate_fretboard_svg(scale_notes, arpeggio_notes, root_note, num_frets=15, box_range=None):
    width, height = 920, 260
    margin_l, margin_r, margin_t = 50, 30, 35
    fret_width = (width - margin_l - margin_r) / num_frets
    string_height = (height - margin_t - 40) / 5
    
    svg = [f'<svg viewBox="0 0 {width} {height}" width="100%" height="auto" style="background:#0f172a; border-radius: 12px; font-family: system-ui, sans-serif; border: 1px solid #334155; display: block;">']
    svg.append(f'<text x="{margin_l}" y="22" fill="#e2e8f0" font-size="13px" font-weight="bold">MÁSTIL (6 Cuerdas - Trastes 0 a {num_frets})</text>')
    svg.append(f'<circle cx="{width - 290}" cy="18" r="6" fill="#f59e0b"/><text x="{width - 278}" y="22" fill="#94a3b8" font-size="11px">Fundamental</text>')
    svg.append(f'<circle cx="{width - 190}" cy="18" r="6" fill="#10b981"/><text x="{width - 178}" y="22" fill="#94a3b8" font-size="11px">Arpegio</text>')
    svg.append(f'<circle cx="{width - 100}" cy="18" r="5" fill="#334155"/><text x="{width - 90}" y="22" fill="#94a3b8" font-size="11px">Escala</text>')

    if box_range:
        b_start, b_end = box_range
        x_start = margin_l + (b_start - 1) * fret_width if b_start > 0 else margin_l
        svg.append(f'<rect x="{x_start}" y="{margin_t - 4}" width="{(margin_l + b_end * fret_width) - x_start}" height="{string_height * 5 + 8}" fill="rgba(56, 189, 248, 0.08)" stroke="#38bdf8" stroke-width="2" stroke-dasharray="4" rx="8" />')

    svg.append(f'<rect x="{margin_l - 6}" y="{margin_t}" width="6" height="{string_height * 5}" fill="#cbd5e1" rx="2" />')
    
    for fret in range(1, num_frets + 1):
        x = margin_l + fret * fret_width
        svg.append(f'<line x1="{x}" y1="{margin_t}" x2="{x}" y2="{margin_t + string_height * 5}" stroke="#475569" stroke-width="2"/>')
        svg.append(f'<text x="{x - fret_width / 2}" y="{height - 12}" fill="#64748b" font-size="11px" font-weight="bold" text-anchor="middle">{fret}</text>')
        
    for fret in (3, 5, 7, 9, 15):
        if fret <= num_frets:
            svg.append(f'<circle cx="{margin_l + (fret - 0.5) * fret_width}" cy="{margin_t + 2.5 * string_height}" r="5" fill="#334155" opacity="0.7"/>')
    if num_frets >= 12:
        cx12 = margin_l + 11.5 * fret_width
        svg.append(f'<circle cx="{cx12}" cy="{margin_t + 1.25 * string_height}" r="4" fill="#334155" opacity="0.7"/>')
        svg.append(f'<circle cx="{cx12}" cy="{margin_t + 3.75 * string_height}" r="4" fill="#334155" opacity="0.7"/>')
        
    for s_idx, open_note in enumerate(GUITAR_TUNING):
        y = margin_t + s_idx * string_height
        svg.append(f'<line x1="{margin_l}" y1="{y}" x2="{width - margin_r}" y2="{y}" stroke="#94a3b8" stroke-width="{1.0 + s_idx * 0.45}"/>')
        svg.append(f'<text x="{margin_l - 18}" y="{y + 4}" fill="#f8fafc" font-size="12px" font-weight="bold" text-anchor="middle">{open_note}</text>')

        open_idx = PITCH_CLASSES.index(open_note)
        for fret in range(0, num_frets + 1):
            note = PITCH_CLASSES[(open_idx + fret) % 12]
            cx = margin_l - 18 if fret == 0 else margin_l + (fret - 0.5) * fret_width
            in_box = (box_range[0] <= fret <= box_range[-1]) if box_range else True
            op = 'opacity="1.0"' if in_box else 'opacity="0.25"'

            if root_note and note == root_note:
                svg.append(f'<g {op}><circle cx="{cx}" cy="{y}" r="11" fill="#f59e0b" stroke="#ffffff" stroke-width="1.5"/><text x="{cx}" y="{y + 4}" fill="#000000" font-size="10px" font-weight="bold" text-anchor="middle">{note}</text></g>')
            elif arpeggio_notes and note in arpeggio_notes:
                svg.append(f'<g {op}><circle cx="{cx}" cy="{y}" r="10" fill="#10b981" stroke="#ffffff" stroke-width="1.2"/><text x="{cx}" y="{y + 4}" fill="#ffffff" font-size="10px" font-weight="bold" text-anchor="middle">{note}</text></g>')
            elif scale_notes and note in scale_notes:
                svg.append(f'<g {op}><circle cx="{cx}" cy="{y}" r="8" fill="#1e293b" stroke="#475569" stroke-width="1"/><text x="{cx}" y="{y + 3.5}" fill="#94a3b8" font-size="9px" text-anchor="middle">{note}</text></g>')
                
    svg.append('</svg>')
    return "".join(svg)

def match_chord(chroma_vector, templates, min_energy=0.01):
    norm = np.linalg.norm(chroma_vector)
    if norm < min_energy:
        return "N"
    scores = {name: np.dot(chroma_vector / norm, t) for name, t in templates.items()}
    best_chord, best_sim = max(scores.items(), key=lambda x: x[-1])
    return best_chord if best_sim >= 0.15 else "N"

st.title("🎸 Jam Companion: Escalas y Acordes")
uploaded_file = st.file_uploader("Sube tu archivo de audio (MP3 o WAV)", type=["mp3", "wav"])

if uploaded_file is not None:
    if st.session_state.get('current_file_name') != uploaded_file.name:
        st.session_state['current_file_name'] = uploaded_file.name
        st.session_state['analysis_done'] = False

    st.audio(uploaded_file)
    
    if st.button("🔍 Analizar Canción"):
        with st.spinner("Analizando armónicos y acordes..."):
            temp_path = f"temp_{uploaded_file.name}"
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            y, sr = librosa.load(temp_path, sr=22050, mono=True)
            duration = float(librosa.get_duration(y=y, sr=sr))
            tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
            tempo_val = float(np.asarray(tempo).flat[0])

            chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_fft=4096, hop_length=1024)
            key_name, key_score = detect_key_accurate(chroma)

            tokens = key_name.split()
            key_root = tokens[0]
            is_major = "Mayor" in key_name
            root_idx = PITCH_CLASSES.index(key_root)

            if is_major:
                scale_notes = [PITCH_CLASSES[(root_idx + s) % 12] for s in (0, 2, 4, 5, 7, 9, 11)]
                penta_notes = [PITCH_CLASSES[(root_idx + s) % 12] for s in (0, 2, 4, 7, 9)]
                rel_key = f"{PITCH_CLASSES[(root_idx + 9) % 12]} menor"
            else:
                scale_notes = [PITCH_CLASSES[(root_idx + s) % 12] for s in (0, 2, 3, 5, 7, 8, 10)]
                penta_notes = [PITCH_CLASSES[(root_idx + s) % 12] for s in (0, 3, 5, 7, 10)]
                rel_key = f"{PITCH_CLASSES[(root_idx + 3) % 12]} Mayor"

            templates = build_smart_chord_templates()
            beat_chroma = librosa.util.sync(chroma, beat_frames, aggregate=np.median)
            beat_times = librosa.frames_to_time(beat_frames, sr=sr, hop_length=1024)
            times = np.concatenate([[0.0], beat_times, [duration]])

            raw_chords = [match_chord(beat_chroma[:, b], templates) for b in range(beat_chroma.shape[-1])]

            table_data, unique_chords = [], []
            if raw_chords:
                curr_chord, start_t = raw_chords[0], times[0]
                for i in range(1, len(raw_chords)):
                    if raw_chords[i] != curr_chord:
                        end_t = times[i]
                        if curr_chord != "N":
                            root, arp_notes = get_arpeggio_details(curr_chord)
                            table_data.append({"Tiempo": f"{start_t:.2f}s - {end_t:.2f}s", "Acorde": curr_chord, "Arpegio": ", ".join(arp_notes)})
                            if curr_chord not in unique_chords: unique_chords.append(curr_chord)
                        curr_chord, start_t = raw_chords[i], end_t
                if curr_chord != "N":
                    root, arp_notes = get_arpeggio_details(curr_chord)
                    table_data.append({"Tiempo": f"{start_t:.2f}s - {duration:.2f}s", "Acorde": curr_chord, "Arpegio": ", ".join(arp_notes)})
                    if curr_chord not in unique_chords: unique_chords.append(curr_chord)

            if os.path.exists(temp_path):
                os.remove(temp_path)

            st.session_state['analysis_done'] = True
            st.session_state['key_name'] = key_name
            st.session_state['key_root'] = key_root
            st.session_state['is_major'] = is_major
            st.session_state['key_score'] = key_score
            st.session_state['tempo_val'] = tempo_val
            st.session_state['scale_notes'] = scale_notes
            st.session_state['penta_notes'] = penta_notes
            st.session_state['rel_key'] = rel_key
            st.session_state['unique_chords'] = unique_chords
            st.session_state['table_data'] = table_data

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

    st.divider()
    st.subheader("🗺️ Mástil de Guitarra y Cajas Pentatónicas")
    col_sel1, col_sel2 = st.columns(2)
    with col_sel1:
        selected_chord = st.selectbox("1. Elige un acorde:", st.session_state['unique_chords'])
    with col_sel2:
        selected_box = st.selectbox("2. Filtrar por Caja Pentatónica:", ["Todo el Mástil (Completo)", "Caja 1", "Caja 2", "Caja 3", "Caja 4", "Caja 5"])

    box_range = None
    if "Caja" in selected_box:
        num = int(selected_box.replace("Caja ", ""))
        box_range = get_pentatonic_box_range(st.session_state['key_root'], st.session_state['is_major'], num)
        st.caption(f"📍 **{selected_box}:** Trastes **{box_range[0]} al {box_range[-1]}**.")

    if selected_chord:
        root_sel, arp_sel = get_arpeggio_details(selected_chord)
        svg = generate_fretboard_svg(st.session_state['scale_notes'], arp_sel, root_sel, num_frets=15, box_range=box_range)
        st.components.v1.html(svg, height=320)

    st.divider()
    st.subheader("📋 Progresión Completa de Acordes")
    st.table(st.session_state['table_data'])
