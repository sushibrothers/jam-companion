import os
import numpy as np
import librosa
import streamlit as st

st.set_page_config(page_title="Jam Companion Pro", page_icon="🎸", layout="wide")

PITCH_CLASSES = tuple(('C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'))
GUITAR_TUNING = tuple(('E', 'B', 'G', 'D', 'A', 'E'))

# Perfiles de Temperley (optimizados para pop, rock y folk como The Beatles)
TEMPERLEY_MAJOR = np.array((5.0, 2.0, 3.5, 2.0, 4.5, 4.0, 2.0, 4.5, 2.0, 3.5, 1.5, 4.0))
TEMPERLEY_MINOR = np.array((5.0, 2.0, 3.5, 4.5, 2.0, 4.0, 2.0, 4.5, 3.5, 2.0, 1.5, 4.0))

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
        maj[(i + 0) % 12] = 1.2; maj[(i + 4) % 12] = 1.0; maj[(i + 7) % 12] = 1.0
        templates[f"{root}"] = maj / np.linalg.norm(maj)

        minor = np.zeros(12)
        minor[(i + 0) % 12] = 1.2; minor[(i + 3) % 12] = 1.0; minor[(i + 7) % 12] = 1.0
        templates[f"{root}m"] = minor / np.linalg.norm(minor)

        dom7 = np.zeros(12)
        dom7[(i + 0) % 12] = 1.2; dom7[(i + 4) % 12] = 1.0; dom7[(i + 7) % 12] = 1.0; dom7[(i + 10) % 12] = 0.8
        templates[f"{root}7"] = dom7 / np.linalg.norm(dom7)
    return templates

def pearson_correlation(x, y):
    x_diff = x - np.mean(x)
    y_diff = y - np.mean(y)
    denom = (np.sqrt(np.sum(x_diff ** 2)) * np.sqrt(np.sum(y_diff ** 2))) + 1e-9
    return float(np.sum(x_diff * y_diff) / denom)

def detect_key_robust(chroma_matrix):
    chroma_mean = np.mean(chroma_matrix, axis=1)
    chroma_mean = chroma_mean / (np.linalg.norm(chroma_mean) + 1e-9)
    
    correlations = {}
    for i, root in enumerate(PITCH_CLASSES):
        maj_prof = np.roll(TEMPERLEY_MAJOR, i)
        min_prof = np.roll(TEMPERLEY_MINOR, i)
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

def get_pentatonic_box_range(key_root, is_major, box_number):
    root_idx = PITCH_CLASSES.index(key_root)
    min_root_idx = (root_idx + 9) % 12 if is_major else root_idx
    base_fret = (min_root_idx - 4) % 12

    offsets = {
        1: (0, 3),
        2: (2, 5),
        3: (5, 8),
        4: (7, 10),
        5: (9, 12),
    }
    start_off, end_off = offsets.get(box_number, (0, 15))
    start_f = (base_fret + start_off) % 12
    end_f = start_f + (end_off - start_off)
    return start_f, end_f

def generate_fretboard_svg(scale_notes, arpeggio_notes, root_note, num_frets=15, box_range=None):
    width, height = 900, 220
    margin_l, margin_r = 50, 30
    margin_t, margin_b = 35, 30
    
    fret_width = (width - margin_l - margin_r) / num_frets
    string_height = (height - margin_t - margin_b) / 5
    
    svg = [f'<svg viewBox="0 0 {width} {height}" width="100%" height="auto" style="background:#0f172a; border-radius: 12px; font-family: system-ui, sans-serif; border: 1px solid #334155;">']
    
    svg.append(f'<text x="{margin_l}" y="22" fill="#e2e8f0" font-size="13px" font-weight="bold">MÁSTIL DE GUITARRA (Trastes 0 a {num_frets})</text>')
    svg.append(f'<circle cx="{width - 290}" cy="18" r="6" fill="#f59e0b"/><text x="{width - 278}" y="22" fill="#94a3b8" font-size="11px">Fundamental</text>')
    svg.append(f'<circle cx="{width - 190}" cy="18" r="6" fill="#10b981"/><text x="{width - 178}" y="22" fill="#94a3b8" font-size="11px">Arpegio</text>')
    svg.append(f'<circle cx="{width - 100}" cy="18" r="5" fill="#334155"/><text x="{width - 90}" y="22" fill="#94a3b8" font-size="11px">Escala</text>')

    if box_range:
        b_start, b_end = box_range
        x_start = margin_l + (b_start - 1) * fret_width if b_start > 0 else margin_l
        x_end = margin_l + b_end * fret_width
        box_w = x_end - x_start
        svg.append(f'<rect x="{x_start}" y="{margin_t - 4}" width="{box_w}" height="{string_height * 5 + 8}" fill="rgba(56, 189, 248, 0.08)" stroke="#38bdf8" stroke-width="2" stroke-dasharray="4" rx="8" />')

    svg.append(f'<rect x="{margin_l - 6}" y="{margin_t}" width="6" height="{string_height * 5}" fill="#cbd5e1" rx="2" />')
    
    for fret in range(1, num_frets + 1):
        x = margin_l + fret * fret_width
        svg.append(f'<line x1="{x}" y1="{margin_t}" x2="{x}" y2="{margin_t + string_height * 5}" stroke="#475569" stroke-width="2"/>')
        svg.append(f'<text x="{x - fret_width / 2}" y="{height - 10}" fill="#64748b" font-size="11px" font-weight="bold" text-anchor="middle">{fret}</text>')
        
    dot_frets = (3, 5, 7, 9, 15)
    for fret in dot_frets:
        if fret <= num_frets:
            cx = margin_l + (fret - 0.5) * fret_width
            cy = margin_t + 2.5 * string_height
            svg.append(f'<circle cx="{cx}" cy="{cy}" r="5" fill="#334155" opacity="0.7"/>')
    if num_frets >= 12:
        cx12 = margin_l + 11.5 * fret_width
        svg.append(f'<circle cx="{cx12}" cy="{margin_t + 1.25 * string_height}" r="4" fill="#334155" opacity="0.7"/>')
        svg.append(f'<circle cx="{cx12}" cy="{margin_t + 3.75 * string_height}" r="4" fill="#334155" opacity="0.7"/>')
        
    for s_idx, open_note in enumerate(GUITAR_TUNING):
        y = margin_t + s_idx * string_height
        thickness = 1.0 + (s_idx * 0.45)
        svg.append(f'<line x1="{margin_l}" y1="{y}" x2="{width - margin_r}" y2="{y}" stroke="#94a3b8" stroke-width="{thickness}"/>')
        svg.append(f'<text x="{margin_l - 18}" y="{y + 4}" fill="#f8fafc" font-size="12px" font-weight="bold" text-anchor="middle">{open_note}</text>')

    for s_idx, open_note in enumerate(GUITAR_TUNING):
        open_idx = PITCH_CLASSES.index(open_note)
        y = margin_t + s_idx * string_height
        for fret in range(0, num_frets + 1):
            note = PITCH_CLASSES[(open_idx + fret) % 12]
            cx = margin_l - 18 if fret == 0 else margin_l + (fret - 0.5) * fret_width
            
            in_box = True
            if box_range:
                b_start, b_end = box_range
                in_box = (b_start <= fret <= b_end)

            opacity_attr = 'opacity="1.0"' if in_box else 'opacity="0.25"'

            if root_note and note == root_note:
                svg.append(f'<g {opacity_attr}><circle cx="{cx}" cy="{y}" r="11" fill="#f59e0b" stroke="#ffffff" stroke-width="1.5"/><text x="{cx}" y="{y + 4}" fill="#000000" font-size="10px" font-weight="bold" text-anchor="middle">{note}</text></g>')
            elif arpeggio_notes and note in arpeggio_notes:
                svg.append(f'<g {opacity_attr}><circle cx="{cx}" cy="{y}" r="10" fill="#10b981" stroke="#ffffff" stroke-width="1.2"/><text x="{cx}" y="{y + 4}" fill="#ffffff" font-size="10px" font-weight="bold" text-anchor="middle">{note}</text></g>')
            elif scale_notes and note in scale_notes:
                svg.append(f'<g {opacity_attr}><circle cx="{cx}" cy="{y}" r="8" fill="#1e293b" stroke="#475569" stroke-width="1"/><text x="{cx}" y="{y + 3.5}" fill="#94a3b8" font-size="9px" text-anchor="middle">{note}</text></g>')
                
    svg.append('</svg>')
    return "".join(svg)

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

st.title("🎸 Jam Companion: Escalas, Acordes y Posiciones Pentatónicas")
st.write("Sube una canción para analizar su tonalidad real, acordes y explorar las 5 posiciones pentatónicas en el mástil.")

uploaded_file = st.file_uploader("Elige un archivo de audio (MP3 o WAV)", type=["mp3", "wav"])

if uploaded_file is not None:
    st.audio(uploaded_file)
    
    if st.button("🔍 Analizar Canción"):
        with st.spinner("Analizando armónicos, afinación y acordes..."):
            temp_path = f"temp_{uploaded_file.name}"
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            y, sr = librosa.load(temp_path, sr=22050, mono=True)
            duration = float(librosa.get_duration(y=y, sr=sr))
            y_harmonic, _ = librosa.effects.hpss(y)

            # Estimación de afinación real
            tuning = librosa.estimate_tuning(y=y_harmonic, sr=sr)

            # Detección de tempo
            tempo, beat_frames = librosa.beat.beat_track(y=y_harmonic, sr=sr)
            tempo_val = float(np.asarray(tempo).flat[0])

            # Cromagrama CENS (suavizado y normalizado) para tonalidad
            chroma_cens = librosa.feature.chroma_cens(y=y_harmonic, sr=sr, tuning=tuning)
            chroma_cqt = librosa.feature.chroma_cqt(y=y_harmonic, sr=sr, tuning=tuning)

            # Detección de tonalidad robusta con perfiles Temperley
            key_name, key_score = detect_key_robust(chroma_cens)

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

            beat_chroma = librosa.util.sync(chroma_cqt, beat_frames, aggregate=np.median)
            beat_times = librosa.frames_to_time(beat_frames, sr=sr)
            times = np.concatenate([[0.0], beat_times, [duration]])

            templates = build_chord_templates()
            num_beats = beat_chroma.shape[-1]
            raw_chords = [match_chord(beat_chroma[:, b], templates) for b in range(num_beats)]

            table_data = []
            unique_chords = []
            if raw_chords:
                curr_chord = raw_chords[0]
                start_t = times[0]
                for i in range(1, len(raw_chords)):
                    if raw_chords[i] != curr_chord:
                        end_t = times[i]
                        if curr_chord != "N":
                            root, arp_notes = get_arpeggio_details(curr_chord)
                            table_data.append({"Tiempo": f"{start_t:.2f}s - {end_t:.2f}s", "Acorde": curr_chord, "Arpegio": ", ".join(arp_notes)})
                            if curr_chord not in unique_chords:
                                unique_chords.append(curr_chord)
                        curr_chord = raw_chords[i]
                        start_t = end_t
                if curr_chord != "N":
                    root, arp_notes = get_arpeggio_details(curr_chord)
                    table_data.append({"Tiempo": f"{start_t:.2f}s - {duration:.2f}s", "Acorde": curr_chord, "Arpegio": ", ".join(arp_notes)})
                    if curr_chord not in unique_chords:
                        unique_chords.append(curr_chord)

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
        selected_chord = st.selectbox("1. Elige un acorde de la canción:", st.session_state['unique_chords'])
    with col_sel2:
        box_options = [
            "Todo el Mástil (Completo)",
            "Caja 1 (Posición Principal)",
            "Caja 2",
            "Caja 3",
            "Caja 4",
            "Caja 5"
        ]
        selected_box = st.selectbox("2. Filtrar por Caja Pentatónica:", box_options)

    box_range = None
    if "Caja 1" in selected_box:
        box_range = get_pentatonic_box_range(st.session_state['key_root'], st.session_state['is_major'], 1)
    elif "Caja 2" in selected_box:
        box_range = get_pentatonic_box_range(st.session_state['key_root'], st.session_state['is_major'], 2)
    elif "Caja 3" in selected_box:
        box_range = get_pentatonic_box_range(st.session_state['key_root'], st.session_state['is_major'], 3)
    elif "Caja 4" in selected_box:
        box_range = get_pentatonic_box_range(st.session_state['key_root'], st.session_state['is_major'], 4)
    elif "Caja 5" in selected_box:
        box_range = get_pentatonic_box_range(st.session_state['key_root'], st.session_state['is_major'], 5)

    if box_range:
        st.caption(f"📍 **{selected_box}:** Enfocada entre los trastes **{box_range[0]} y {box_range}**.")

    if selected_chord:
        root_sel, arp_sel = get_arpeggio_details(selected_chord)
        svg_code = generate_fretboard_svg(
            st.session_state['scale_notes'], 
            arp_sel, 
            root_sel, 
            num_frets=15, 
            box_range=box_range
        )
        st.components.v1.html(svg_code, height=240)

    st.divider()
    st.subheader("📋 Progresión Completa de Acordes")
    st.table(st.session_state['table_data'])
