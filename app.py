import os
import numpy as np
import librosa
import streamlit as st

st.set_page_config(page_title="Jam Companion Pro", page_icon="🎸", layout="wide")

PITCH_CLASSES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
GUITAR_TUNING = ['E', 'B', 'G', 'D', 'A', 'E']

MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

CHORD_DEFINITIONS = {
    "": ("Mayor",),
    "m": ("menor",),
    "7": ("7ma Dominante",),
    "maj7": ("Mayor 7ma",),
    "m7": ("menor 7ma",),
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

def detect_key(chroma_mean):
    correlations = {}
    for i, root in enumerate(PITCH_CLASSES):
        maj_prof = np.roll(MAJOR_PROFILE, i)
        min_prof = np.roll(MINOR_PROFILE, i)
        correlations[f"{root} Mayor"] = np.corrcoef(chroma_mean, maj_prof)
        correlations[f"{root} menor"] = np.corrcoef(chroma_mean, min_prof)
    best_key, score = max(correlations.items(), key=lambda x: x)
    return best_key, score

def parse_chord(chord_str):
    if chord_str == "N" or not chord_str:
        return None, None
    if len(chord_str) > 1 and chord_str == "#":
        return chord_str[:2], chord_str[2:]
    return chord_str[0], chord_str[1:]

def get_arpeggio_details(chord_str):
    root, quality = parse_chord(chord_str)
    if not root or root not in PITCH_CLASSES:
        return None, []
    root_idx = PITCH_CLASSES.index(root)
    name_qual, intervals = CHORD_DEFINITIONS.get(quality, CHORD_DEFINITIONS[""])
    notes = [PITCH_CLASSES[(root_idx + interval) % 12] for interval in intervals]
    return root, notes

def generate_fretboard_svg(scale_notes, arpeggio_notes, root_note, num_frets=12):
    width, height = 850, 220
    margin_l, margin_r = 50, 30
    margin_t, margin_b = 35, 30
    
    fret_width = (width - margin_l - margin_r) / num_frets
    string_height = (height - margin_t - margin_b) / 5
    
    svg = [f'<svg viewBox="0 0 {width} {height}" width="100%" height="auto" style="background:#0f172a; border-radius: 12px; font-family: system-ui, sans-serif; border: 1px solid #334155;">']
    
    svg.append(f'<text x="{margin_l}" y="22" fill="#e2e8f0" font-size="13px" font-weight="bold">MÁSTIL DE GUITARRA (Trastes 0 a {num_frets})</text>')
    svg.append(f'<circle cx="{width - 290}" cy="18" r="6" fill="#f59e0b"/><text x="{width - 278}" y="22" fill="#94a3b8" font-size="11px">Fundamental</text>')
    svg.append(f'<circle cx="{width - 190}" cy="18" r="6" fill="#10b981"/><text x="{width - 178}" y="22" fill="#94a3b8" font-size="11px">Arpegio</text>')
    svg.append(f'<circle cx="{width - 100}" cy="18" r="5" fill="#334155"/><text x="{width - 90}" y="22" fill="#94a3b8" font-size="11px">Escala</text>')

    svg.append(f'<rect x="{margin_l - 6}" y="{margin_t}" width="6" height="{string_height * 5}" fill="#cbd5e1" rx="2" />')
    
    for fret in range(1, num_frets + 1):
        x = margin_l + fret * fret_width
        svg.append(f'<line x1="{x}" y1="{margin_t}" x2="{x}" y2="{margin_t + string_height * 5}" stroke="#475569" stroke-width="2"/>')
        svg.append(f'<text x="{x - fret_width / 2}" y="{height - 10}" fill="#64748b" font-size="11px" font-weight="bold" text-anchor="middle">{fret}</text>')
        
    for fret in:
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
            if root_note and note == root_note:
                svg.append(f'<circle cx="{cx}" cy="{y}" r="11" fill="#f59e0b" stroke="#ffffff" stroke-width="1.5"/>')
                svg.append(f'<text x="{cx}" y="{y + 4}" fill="#000000" font-size="10px" font-weight="bold" text-anchor="middle">{note}</text>')
            elif arpeggio_notes and note in arpeggio_notes:
                svg.append(f'<circle cx="{cx}" cy="{y}" r="10" fill="#10b981" stroke="#ffffff" stroke-width="1.2"/>')
                svg.append(f'<text x="{cx}" y="{y + 4}" fill="#ffffff" font-size="10px" font-weight="bold" text-anchor="middle">{note}</text>')
            elif scale_notes and note in scale_notes:
                svg.append(f'<circle cx="{cx}" cy="{y}" r="8" fill="#1e293b" stroke="#475569" stroke-width="1"/>')
                svg.append(f'<text x="{cx}" y="{y + 3.5}" fill="#94a3b8" font-size="9px" text-anchor="middle">{note}</text>')
                
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

# Interfaz Principal
st.title("🎸 Jam Companion: Escalas, Acordes y Mástil")
st.write("Sube una canción para analizar su tonalidad, acordes y ver el mapa de digitación en el mástil.")

uploaded_file = st.file_uploader("Elige un archivo de audio (MP3 o WAV)", type=["mp3", "wav"])

if uploaded_file is not None:
    st.audio(uploaded_file)
    
    if st.button("🔍 Analizar Canción"):
        with st.spinner("Analizando armónicos y acordes con librosa..."):
            temp_path = f"temp_{uploaded_file.name}"
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            y, sr = librosa.load(temp_path, sr=22050, mono=True)
            duration = float(librosa.get_duration(y=y, sr=sr))
            y_harmonic, _ = librosa.effects.hpss(y)

            tempo, beat_frames = librosa.beat.beat_track(y=y_harmonic, sr=sr)
            tempo_val = float(np.atleast_1d(tempo)[0])

            chroma = librosa.feature.chroma_cqt(y=y_harmonic, sr=sr)
            chroma_mean = np.mean(chroma, axis=1)
            key_name, key_score = detect_key(chroma_mean)

            tokens = key_name.split()
            key_root = tokens[0]
            is_major = "Mayor" in key_name
            root_idx = PITCH_CLASSES.index(key_root)

            if is_major:
                scale_notes = [PITCH_CLASSES[(root_idx + s) % 12] for s in]
                penta_notes = [PITCH_CLASSES[(root_idx + s) % 12] for s in]
                rel_key = f"{PITCH_CLASSES[(root_idx + 9) % 12]} menor"
            else:
                scale_notes = [PITCH_CLASSES[(root_idx + s) % 12] for s in]
                penta_notes = [PITCH_CLASSES[(root_idx + s) % 12] for s in]
                rel_key = f"{PITCH_CLASSES[(root_idx + 3) % 12]} Mayor"

            beat_chroma = librosa.util.sync(chroma, beat_frames, aggregate=np.median)
            beat_times = librosa.frames_to_time(beat_frames, sr=sr)
            times = np.concatenate([[0.0], beat_times, [duration]])

            templates = build_chord_templates()
            raw_chords = [match_chord(beat_chroma[:, b], templates) for b in range(beat_chroma.shape)]

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

    st.subheader("🗺️ Mástil de Guitarra (Fretboard)")
    selected = st.selectbox("Selecciona un acorde para ver sus notas en el mástil:", st.session_state['unique_chords'])
    
    if selected:
        root_sel, arp_sel = get_arpeggio_details(selected)
        svg_code = generate_fretboard_svg(st.session_state['scale_notes'], arp_sel, root_sel)
        st.components.v1.html(svg_code, height=230)

    st.subheader("📋 Progresión de Acordes")
    st.table(st.session_state['table_data'])
