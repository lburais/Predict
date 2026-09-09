#!/usr/bin/env python3
"""
gpx_power.py — Calcule la puissance ou la vitesse à partir d'un fichier GPX.

Modele physique :
    W  = Wf + Wa + Wg                          (puissance totale, en watts)
    Wg = V x F x (p / 100)                     (puissance due a la pente)
    Wa = 0.5 x mva x SCx x V^3                 (puissance aerodynamique)
    Wf = Kf x V x F                            (puissance de frottement)

    V   = vitesse (m/s)
    F   = poids en kg x g (g = 9.81)           (force, en Newton)
    p   = pente (%)
    SCx = coefficient aerodynamique (defaut 0.40)
    mva = masse volumique de l'air (defaut 1.2)
    Kf  = coefficient de frottement (defaut 0.01)

Deux modes :
    - power : le GPX contient des horodatages (time). Le programme calcule
              V a partir de la distance/temps entre points, puis en deduit
              la puissance necessaire pour un poids donne.
    - speed : le programme calcule, pour un poids et une puissance moyenne
              donnes, la vitesse V sur chaque segment (a partir de la pente
              donnee par les altitudes du GPX), en resolvant l'equation
              cubique : 0.5*mva*SCx*V^3 + (Kf*F + F*p/100)*V - W = 0
"""

import argparse
import math
import os
import sys
import xml.etree.ElementTree as ET
from datetime import datetime
from io import BytesIO

try:
    from matplotlib import pyplot as plt
except ImportError:  # pragma: no cover
    plt = None

try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
except ImportError:  # pragma: no cover
    colors = None
    letter = None
    getSampleStyleSheet = None
    Paragraph = None
    SimpleDocTemplate = None
    Spacer = None
    Table = None
    TableStyle = None

G = 9.81


# --------------------------------------------------------------------------
# Lecture du GPX
# --------------------------------------------------------------------------

def _localname(tag):
    """Retire le namespace XML d'un tag ('{ns}trkpt' -> 'trkpt')."""
    return tag.split('}', 1)[1] if '}' in tag else tag


def parse_gpx(path):
    """Retourne une liste de points {lat, lon, ele, time} dans l'ordre du track."""
    tree = ET.parse(path)
    root = tree.getroot()

    points = []
    for trkpt in root.iter():
        if _localname(trkpt.tag) != 'trkpt':
            continue
        lat = float(trkpt.attrib['lat'])
        lon = float(trkpt.attrib['lon'])
        ele = None
        time = None
        for child in trkpt:
            name = _localname(child.tag)
            if name == 'ele' and child.text:
                ele = float(child.text)
            elif name == 'time' and child.text:
                time = datetime.fromisoformat(child.text.replace('Z', '+00:00'))
        points.append({'lat': lat, 'lon': lon, 'ele': ele, 'time': time})

    if len(points) < 2:
        sys.exit("Erreur : le fichier GPX contient moins de 2 points de trace.")
    return points


def haversine(lat1, lon1, lat2, lon2):
    """Distance en metres entre deux points GPS."""
    R = 6371000.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def build_segments(points):
    """Construit la liste des segments avec distance (m), pente (%)
    et duree (s) si les horodatages sont presents."""
    segments = []
    for p1, p2 in zip(points, points[1:]):
        dist = haversine(p1['lat'], p1['lon'], p2['lat'], p2['lon'])
        if dist == 0:
            continue  # points identiques, on ignore

        if p1['ele'] is not None and p2['ele'] is not None:
            dele = p2['ele'] - p1['ele']
            pente = (dele / dist) * 100.0
        else:
            pente = 0.0

        dt = None
        if p1['time'] is not None and p2['time'] is not None:
            dt = (p2['time'] - p1['time']).total_seconds()

        segments.append({'dist': dist, 'pente': pente, 'dt': dt})
    return segments


def build_segment_groups(segments, target_distance=100.0):
    """Regroupe des segments consecutifs en tranches dont la distance est la
    plus proche possible au-dessus de target_distance."""
    groups = []
    current_segments = []
    current_dist = 0.0

    def finalize_group(segment_list, total_dist):
        if not segment_list:
            return None
        return {
            'segments': segment_list,
            'dist': total_dist,
            'pente': (
                sum(seg['pente'] * seg['dist'] for seg in segment_list) / total_dist
                if total_dist > 0 else 0.0
            ),
            'dt': (
                sum(seg['dt'] for seg in segment_list if seg['dt'] is not None)
                if any(seg['dt'] is not None for seg in segment_list) else None
            ),
        }

    for seg in segments:
        seg_dist = seg.get('dist', 0.0)
        if seg_dist <= 0:
            continue

        if current_dist + seg_dist < target_distance:
            current_segments.append(seg)
            current_dist += seg_dist
            continue

        current_segments.append(seg)
        current_dist += seg_dist
        group = finalize_group(current_segments, current_dist)
        if group is not None:
            groups.append(group)
        current_segments = []
        current_dist = 0.0

    if current_segments:
        group = finalize_group(current_segments, current_dist)
        if group is not None:
            groups.append(group)

    return groups


# --------------------------------------------------------------------------
# Physique
# --------------------------------------------------------------------------

def total_weight(cyclist, bike=9.0, accessories=3.0):
    """Retourne le poids total du cycliste, du velo et des accessoires."""
    cyclist_weight = cyclist if cyclist is not None else 70.0
    return cyclist_weight + bike + accessories


def puissance(v, poids_kg, pente_pct, scx, mva, kf):
    """Calcule W a partir de V (m/s), du poids (kg) et de la pente (%)."""
    F = poids_kg * G
    wf = kf * v * F
    wa = 0.5 * mva * scx * v ** 3
    wg = v * F * (pente_pct / 100.0)
    return wf + wa + wg


def compute_weighted_average_power(groups, weight, min_slope=0.0, scx=0.40, mva=1.2, kf=0.01):
    """Calcule la puissance moyenne ponderee par le temps en ne considerant
    que les groupes dont la pente est superieure ou egale a min_slope."""
    filtered_groups = [
        grp for grp in groups
        if grp.get('pente', 0.0) >= min_slope and grp.get('dt', 0) is not None and grp.get('dt', 0) > 0
    ]
    if not filtered_groups:
        return 0.0

    total_w = 0.0
    total_dt = 0.0
    for grp in filtered_groups:
        v = grp['dist'] / grp['dt']
        w = max(puissance(v, weight, grp['pente'], scx, mva, kf), 0.0)
        if w <= 0.0:
            continue
        total_w += w * grp['dt']
        total_dt += grp['dt']

    return total_w / total_dt if total_dt > 0 else 0.0


def compute_speed_comparison(groups, power, cyclist, bike=9.0, accessories=3.0, min_slope=0.0, scx=0.40, mva=1.2, kf=0.01, chainrings=None, cogs=None, rpm_min=60.0, rpm_max=80.0, max_cog=None):
    """Retourne les vitesses GPX et estimees par groupe, ainsi que les temps totaux.
    Si un groupe n'a pas d'horodatage, la vitesse GPX et le delta sont omis."""
    results = []
    total_dist = 0.0
    total_gpx_time = 0.0
    total_est_time = 0.0
    filtered_dist = 0.0
    filtered_gpx_time = 0.0
    filtered_est_time = 0.0
    weight = total_weight(cyclist, bike=bike, accessories=accessories)

    for grp in groups:
        dt = grp.get('dt')
        dist = grp.get('dist', 0.0)
        if dist <= 0:
            continue

        estimated_speed = vitesse(power, weight, grp['pente'], scx, mva, kf)
        estimated_duration = (dist / estimated_speed) if estimated_speed > 0 else 0.0
        result = {
            'dist': dist,
            'pente': grp['pente'],
            'estimated_speed': estimated_speed,
            'estimated_duration': estimated_duration,
            'gpx_speed': None,
            'delta': None,
            'gear': recommend_gear(
                estimated_speed,
                chainrings=chainrings,
                cogs=cogs,
                rpm_min=rpm_min,
                rpm_max=rpm_max,
                max_cog=max_cog,
            ),
        }

        if dt is not None and dt > 0:
            gpx_speed = dist / dt
            result['gpx_speed'] = gpx_speed
            result['delta'] = estimated_speed - gpx_speed
            total_gpx_time += dt
            total_dist += dist
            if estimated_speed > 0:
                total_est_time += dist / estimated_speed

            if grp.get('pente', 0.0) >= min_slope:
                filtered_dist += dist
                filtered_gpx_time += dt
                if estimated_speed > 0:
                    filtered_est_time += dist / estimated_speed
        else:
            total_dist += dist
            if estimated_speed > 0:
                total_est_time += dist / estimated_speed
            if grp.get('pente', 0.0) >= min_slope:
                filtered_dist += dist
                if estimated_speed > 0:
                    filtered_est_time += dist / estimated_speed

        results.append(result)

    return results, total_dist, total_gpx_time, total_est_time, filtered_dist, filtered_gpx_time, filtered_est_time


def format_gear_label(gear):
    """Formate une recommandation de transmission sous la forme 'chainring x cog'."""
    if not gear:
        return ''
    return f"{gear.get('chainring', '')} x {gear.get('cog', '')}"


def format_duration_hms(total_seconds):
    """Formate une duree en h mn s (ex. 1h 02mn 03s)."""
    seconds = max(int(total_seconds), 0)
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours}h {minutes:02d}mn {seconds:02d}s"


def recommend_gear(speed_mps, chainrings=None, cogs=None, rpm_min=60.0, rpm_max=80.0, max_cog=None):
    """Retourne le meilleur couple chaine/plateau pour atteindre une vitesse donnee dans la plage de cadence desiree."""
    chainrings = chainrings or [50, 34]
    cogs = cogs or [14, 16, 18, 20, 22, 24, 26, 28, 32, 36]
    if not chainrings or not cogs:
        return None

    wheel_circumference = 3.2
    best = None
    best_score = None

    for chainring in chainrings:
        for cog in cogs:
            ratio = chainring / cog
            cadence_rpm = (speed_mps * 60.0) / (wheel_circumference * ratio)
            if cadence_rpm <= 0:
                continue

            largest_chainring = max(chainrings)
            is_cross_chain = (
                chainring == largest_chainring
                and max_cog is not None
                and cog > max_cog
            )
            if is_cross_chain:
                continue

            within_range = rpm_min <= cadence_rpm <= rpm_max
            target_rpm = (rpm_min + rpm_max) / 2.0
            distance_to_range = 0.0 if within_range else min(abs(cadence_rpm - rpm_min), abs(cadence_rpm - rpm_max))
            cadence_penalty = abs(cadence_rpm - target_rpm) if within_range else distance_to_range + 10.0
            score = (0.0 if within_range else 1.0, cadence_penalty, abs(chainring - max(chainrings)))
            if best_score is None or score < best_score:
                best = {
                    'chainring': chainring,
                    'cog': cog,
                    'ratio': ratio,
                    'cadence_rpm': cadence_rpm,
                    'within_range': within_range,
                }
                best_score = score

    if best is None:
        for chainring in chainrings:
            for cog in cogs:
                ratio = chainring / cog
                cadence_rpm = (speed_mps * 60.0) / (wheel_circumference * ratio)
                if cadence_rpm <= 0:
                    continue
                best = {
                    'chainring': chainring,
                    'cog': cog,
                    'ratio': ratio,
                    'cadence_rpm': cadence_rpm,
                    'within_range': False,
                }
                break
            if best is not None:
                break

    return best


def vitesse(w_cible, poids_kg, pente_pct, scx, mva, kf):
    """Resout 0.5*mva*scx*V^3 + (kf*F + F*pente/100)*V - W = 0 pour V (m/s).

    Utilise la methode de Cardano pour trouver la racine reelle positive
    d'un polynome cubique a x^3 + b x^2 + c x + d = 0 (ici b = 0).
    """
    F = poids_kg * G
    a = 0.5 * mva * scx
    c = kf * F + F * (pente_pct / 100.0)
    d = -w_cible

    if a == 0:
        # Cas degenere : equation lineaire c*V + d = 0
        return -d / c if c != 0 else 0.0

    # Forme deprimee : t^3 + p*t + q = 0, avec x = t (car b = 0)
    p = c / a
    q = d / a

    disc = (q / 2) ** 2 + (p / 3) ** 3

    if disc >= 0:
        sqrt_disc = math.sqrt(disc)
        u = math.copysign(abs(-q / 2 + sqrt_disc) ** (1 / 3), -q / 2 + sqrt_disc)
        v_ = math.copysign(abs(-q / 2 - sqrt_disc) ** (1 / 3), -q / 2 - sqrt_disc)
        real_root = u + v_
        return max(real_root, 0.0)
    else:
        # Trois racines reelles : on prend la positive pertinente
        r = math.sqrt(-(p ** 3) / 27)
        phi = math.acos(max(-1.0, min(1.0, -q / (2 * r))))
        roots = [2 * (r ** (1 / 3)) * math.cos((phi + 2 * math.pi * k) / 3) for k in range(3)]
        positive_roots = [rt for rt in roots if rt > 0]
        return max(positive_roots) if positive_roots else 0.0


# --------------------------------------------------------------------------
# Modes
# --------------------------------------------------------------------------

def iter_gpx_files(folder=None):
    """Retourne la liste ordonnee des fichiers GPX d'un dossier."""
    folder = folder or '.'
    if os.path.isfile(folder):
        return [folder] if folder.lower().endswith('.gpx') else []
    if not os.path.isdir(folder):
        return []
    return sorted(
        os.path.join(folder, name)
        for name in os.listdir(folder)
        if name.lower().endswith('.gpx')
    )


def build_pdf_output_path(gpx_path, mode=None):
    """Construit le chemin PDF par defaut a partir du chemin GPX."""
    if not gpx_path:
        return None
    base, ext = os.path.splitext(gpx_path)
    suffix = f" ({mode})" if mode else ""
    if ext.lower() == '.gpx':
        return f"{base}{suffix}.pdf"
    return f"{gpx_path}{suffix}.pdf"


def export_results_pdf(output_path, rows, title, chart_rows=None, summary=None):
    """Exporte un tableau de resultats en PDF avec un graphique, un resume et le tableau."""
    if not all([SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, colors, letter, getSampleStyleSheet]):
        raise RuntimeError("reportlab n'est pas disponible")

    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(output_path, pagesize=letter)
    elements = [Paragraph(title, styles['Title']), Spacer(1, 12)]

    if chart_rows:
        chart_buffer = BytesIO()
        export_speed_chart(chart_buffer, chart_rows, fmt='png')
        chart_buffer.seek(0)
        if chart_buffer.getbuffer().nbytes > 0:
            elements.append(Image(chart_buffer, width=450, height=220))
            elements.append(Spacer(1, 12))

    if summary:
        elements.append(Paragraph('Resume', styles['Heading2']))
        for line in summary:
            elements.append(Paragraph(line, styles['BodyText']))
        elements.append(Spacer(1, 12))

    if 'vitesse' in title.lower():
        table_data = [['group', 'dist(km)', 'cum_dist(km)', 'pente(%)', 'GPX(km/h)', 'Estimated(km/h)', 'delta(km/h)', 'gear', 'cadence(rpm)']]
        cumulative_distance = 0.0
        for row in rows:
            cumulative_distance += row.get('dist', 0.0)
            gear = row.get('gear') or {}
            gpx_speed = row.get('gpx_speed')
            if gpx_speed is not None:
                gpx_speed_kmh = gpx_speed * 3.6
            else:
                gpx_speed_kmh = None
            estimated_speed_kmh = row.get('estimated_speed', 0.0) * 3.6
            delta = row.get('delta')
            if delta is not None:
                delta_kmh = delta * 3.6
            else:
                delta_kmh = None
            table_data.append([
                row['group'],
                f"{row['dist']/1000.0:.3f}",
                f"{cumulative_distance/1000.0:.3f}",
                f"{row['pente']:.2f}",
                '-' if gpx_speed_kmh is None else f"{gpx_speed_kmh:.2f}",
                f"{estimated_speed_kmh:.2f}",
                '-' if delta_kmh is None else f"{delta_kmh:.2f}",
                format_gear_label(gear),
                f"{gear.get('cadence_rpm', 0.0):.0f}",
            ])
    else:
        table_data = [['group', 'dist(km)', 'cum_dist(km)', 'pente(%)', 'V(km/h)', 'P(W)', 'gear', 'cadence(rpm)']]
        cumulative_distance = 0.0
        for row in rows:
            cumulative_distance += row.get('dist', 0.0)
            speed = row.get('speed', row.get('estimated_speed', 0.0)) * 3.6
            power = max(row.get('power', row.get('estimated_speed', 0.0)), 0.0)
            gear = row.get('gear') or {}
            table_data.append([
                row['group'],
                f"{row['dist']/1000.0:.3f}",
                f"{cumulative_distance/1000.0:.3f}",
                f"{row['pente']:.2f}",
                f"{speed:.2f}",
                f"{power:.1f}",
                format_gear_label(gear),
                f"{gear.get('cadence_rpm', 0.0):.0f}",
            ])

    table = Table(table_data, repeatRows=1)
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
    ]))
    elements.append(table)
    doc.build(elements)


def export_speed_chart(output_path, rows, fmt='png'):
    """Exporte un graphique de la vitesse par distance cumulative si matplotlib est disponible."""
    if plt is None:
        raise RuntimeError("matplotlib n'est pas disponible")

    x = []
    y = []
    cumulative_distance = 0.0
    for row in rows:
        cumulative_distance += row.get('dist', 0.0)
        x.append(cumulative_distance)
        if 'estimated_speed' in row:
            y.append(row['estimated_speed'] * 3.6)
        elif 'speed' in row:
            y.append(row['speed'] * 3.6)
        elif 'gpx_speed' in row:
            y.append(row['gpx_speed'] * 3.6)
        elif 'power' in row:
            y.append(0.0)
        else:
            y.append(0.0)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(x, y, marker='o', linewidth=1.5)
    ax.set_title('Vitesse par distance cumulative')
    ax.set_xlabel('Distance cumulative (m)')
    ax.set_ylabel('Vitesse (km/h)')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    if hasattr(output_path, 'write'):
        fig.savefig(output_path, format=fmt, dpi=150)
    else:
        fig.savefig(output_path, dpi=150)
    plt.close(fig)


def mode_power(args):
    points = parse_gpx(args.gpx)
    segments = build_segments(points)
    groups = build_segment_groups(segments, target_distance=args.group_distance)

    total_w = 0.0
    total_dt = 0.0
    n = 0

    rows = []
    cumulative_distance = 0.0
    print(f"{'group':>8} {'dist(km)':>11} {'cum_dist(km)':>13} {'pente(%)':>11} {'V(m/s)':>11} {'V(km/h)':>13} {'P(W)':>9} {'gear':>12} {'cadence(rpm)':>13}")
    for i, grp in enumerate(groups, 1):
        if grp['dt'] is None or grp['dt'] <= 0:
            continue  # pas d'horodatage exploitable pour ce groupe
        v = grp['dist'] / grp['dt']
        total_mass = total_weight(args.cyclist, bike=args.bike, accessories=args.accessories)
        w = puissance(v, total_mass, grp['pente'], args.scx, args.mva, args.kf)
        cumulative_distance += grp['dist']
        w = max(w, 0.0)
        gear = recommend_gear(
            v,
            chainrings=args.chainrings,
            cogs=args.cogs,
            rpm_min=args.rpm_range[0],
            rpm_max=args.rpm_range[1],
            max_cog=args.max_cog,
        )
        print(f"{i:8d} {grp['dist']/1000.0:10.3f}km {cumulative_distance/1000.0:12.3f}km {grp['pente']:10.2f}% {v:8.2f}m/s {v * 3.6:9.2f}km/h {w:8.1f}W {format_gear_label(gear):>12} {gear.get('cadence_rpm', 0.0):>10.0f}")
        rows.append({
            'group': i,
            'dist': grp['dist'],
            'pente': grp['pente'],
            'speed': v,
            'power': w,
            'gear': gear,
        })
        total_w += w * grp['dt']
        total_dt += grp['dt']
        n += 1

    if n == 0:
        # sys.exit("Erreur : aucun horodatage (balise <time>) exploitable dans ce GPX pour le mode 'power'.")
        print(f"\nErreur : aucun horodatage (balise <time>) exploitable dans ce GPX pour le mode 'power'.")
        return

    total_mass = total_weight(args.cyclist, bike=args.bike, accessories=args.accessories)
    average_power = compute_weighted_average_power(
        groups,
        total_mass,
        min_slope=args.min_slope,
        scx=args.scx,
        mva=args.mva,
        kf=args.kf,
    )
    print(f"\nPuissance moyenne (ponderee par le temps sur pente >= {args.min_slope:.2f}%) : {average_power:.1f} W sur {n} groupes")

    summary_lines = [
        "Parametres",
        f"Cycliste : {args.cyclist if args.cyclist is not None else 70.0:.1f} kg",
        f"Velo : {args.bike:.1f} kg",
        f"Accessoires : {args.accessories:.1f} kg",
        f"Distance cible : {args.group_distance:.0f} m",
        f"Pente minimale : {args.min_slope:.2f}%",
        f"Plateaux : {', '.join(str(value) for value in args.chainrings)}",
        f"Pignons : {', '.join(str(value) for value in args.cogs)}",
        f"Cadence : {args.rpm_range[0]:.0f}-{args.rpm_range[1]:.0f} rpm",
        f"Pignon max pour grand plateau : {args.max_cog if args.max_cog is not None else 'non defini'}",
        f"SCx : {args.scx:.2f} | mva : {args.mva:.2f} | kf : {args.kf:.2f}",
    ]
    for line in summary_lines:
        print(line)

    if args.gpx:
        output_path = build_pdf_output_path(args.gpx, mode=args.mode if getattr(args, 'batch_mode', False) else None)
        output_dir = os.path.dirname(output_path) or '.'
        os.makedirs(output_dir, exist_ok=True)
        try:
            export_results_pdf(
                output_path,
                rows,
                'Tableau puissance par distance',
                chart_rows=rows,
                summary=[
                    *summary_lines,
                    f"Distance totale : {cumulative_distance/1000.0:.3f} km",
                    f"Nombre de groupes : {n}",
                    f"Puissance moyenne : {average_power:.1f} W",
                ],
            )
            print(f"PDF exporte : {output_path}")
        except RuntimeError as exc:
            print(f"Impossible d'exporter le PDF : {exc}")


def mode_speed(args):
    points = parse_gpx(args.gpx)
    segments = build_segments(points)
    groups = build_segment_groups(segments, target_distance=args.group_distance)

    results, total_dist, total_gpx_time, total_est_time, filtered_dist, filtered_gpx_time, filtered_est_time = compute_speed_comparison(
        groups,
        power=args.power,
        cyclist=args.cyclist,
        bike=args.bike,
        accessories=args.accessories,
        min_slope=args.min_slope,
        scx=args.scx,
        mva=args.mva,
        kf=args.kf,
        chainrings=args.chainrings,
        cogs=args.cogs,
        rpm_min=args.rpm_range[0],
        rpm_max=args.rpm_range[1],
        max_cog=args.max_cog,
    )

    cumulative_distance = 0.0
    print(f"{'group':>8} {'dist(km)':>11} {'cum_dist(km)':>13} {'pente(%)':>11} {'GPX(km/h)':>12} {'Estimated(km/h)':>17} {'delta(km/h)':>13} {'gear':>12} {'cadence(rpm)':>13}")
    for i, res in enumerate(results, 1):
        cumulative_distance += res['dist']
        gear = res.get('gear') or {}
        gpx_speed = res.get('gpx_speed')
        delta = res.get('delta')
        gpx_value = '-' if gpx_speed is None else f"{gpx_speed * 3.6:8.2f}"
        delta_value = '-' if delta is None else f"{delta * 3.6:9.2f}"
        print(
            f"{i:8d} {res['dist']/1000.0:10.3f}km {cumulative_distance/1000.0:12.3f}km {res['pente']:10.2f}% {gpx_value:>10} {res['estimated_speed'] * 3.6:14.2f} {delta_value:>10} {format_gear_label(gear):>12} {gear.get('cadence_rpm', 0.0):>10.0f}"
        )

    if total_est_time > 0:
        est_avg = total_dist / total_est_time if total_est_time > 0 else 0.0
        print(f"\nVitesse moyenne estimee : {est_avg * 3.6:.2f} km/h")
        print(f"Duree estimee totale : {format_duration_hms(total_est_time)}")
        if total_gpx_time > 0:
            gpx_avg = total_dist / total_gpx_time if total_gpx_time > 0 else None
            print(f"Vitesse moyenne GPX : {gpx_avg * 3.6:.2f} km/h")
            print(f"Difference moyenne : {(est_avg - gpx_avg) * 3.6:.2f} km/h")

    if filtered_est_time > 0:
        filtered_est_avg = filtered_dist / filtered_est_time if filtered_est_time > 0 else 0.0
        print(f"\nVitesse moyenne estimee sur pente >= {args.min_slope:.2f}% : {filtered_est_avg * 3.6:.2f} km/h")
        if filtered_gpx_time > 0:
            filtered_gpx_avg = filtered_dist / filtered_gpx_time if filtered_gpx_time > 0 else None
            print(f"Vitesse moyenne GPX sur pente >= {args.min_slope:.2f}% : {filtered_gpx_avg * 3.6:.2f} km/h")
            print(f"Difference moyenne sur pente >= {args.min_slope:.2f}% : {(filtered_est_avg - filtered_gpx_avg) * 3.6:.2f} km/h")

    summary_lines = [
        "Parametres",
        f"Cycliste : {args.cyclist if args.cyclist is not None else 70.0:.1f} kg",
        f"Velo : {args.bike:.1f} kg",
        f"Accessoires : {args.accessories:.1f} kg",
        f"Puissance : {args.power:.1f} W",
        f"Distance cible : {args.group_distance:.0f} m",
        f"Pente minimale : {args.min_slope:.2f}%",
        f"Plateaux : {', '.join(str(value) for value in args.chainrings)}",
        f"Pignons : {', '.join(str(value) for value in args.cogs)}",
        f"Cadence : {args.rpm_range[0]:.0f}-{args.rpm_range[1]:.0f} rpm",
        f"Pignon max pour grand plateau : {args.max_cog if args.max_cog is not None else 'non defini'}",
        f"SCx : {args.scx:.2f} | mva : {args.mva:.2f} | kf : {args.kf:.2f}",
    ]
    for line in summary_lines:
        print(line)

    if args.gpx:
        output_path = build_pdf_output_path(args.gpx, mode=args.mode if getattr(args, 'batch_mode', False) else None)
        output_dir = os.path.dirname(output_path) or '.'
        os.makedirs(output_dir, exist_ok=True)
        try:
            summary_items = [
                *summary_lines,
                f"Distance totale : {total_dist/1000.0:.3f} km",
                f"Duree estimee totale : {format_duration_hms(total_est_time)}",
            ]
            if total_gpx_time > 0:
                summary_items.extend([
                    f"Temps GPX : {total_gpx_time:.1f} s",
                    f"Vitesse moyenne GPX : {((total_dist / total_gpx_time) if total_gpx_time > 0 else 0.0) * 3.6:.2f} km/h",
                ])
            export_results_pdf(
                output_path,
                [
                    {
                        'group': i,
                        'dist': res['dist'],
                        'pente': res['pente'],
                        'gpx_speed': res.get('gpx_speed'),
                        'estimated_speed': res['estimated_speed'],
                        'estimated_duration': res.get('estimated_duration', 0.0),
                        'delta': res.get('delta'),
                        'gear': res.get('gear'),
                    } for i, res in enumerate(results, 1)
                ],
                'Tableau vitesse par groupe',
                chart_rows=[{'dist': res['dist'], 'estimated_speed': res['estimated_speed'], 'gpx_speed': res.get('gpx_speed')} for res in results],
                summary=summary_items,
            )
            print(f"PDF exporte : {output_path}")
        except RuntimeError as exc:
            print(f"Impossible d'exporter le PDF : {exc}")


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Calcule puissance ou vitesse a partir d'un fichier GPX.")
    parser.add_argument('mode', nargs='?', choices=['power', 'speed'], help="power : calcule la puissance ; speed : calcule la vitesse")
    parser.add_argument('--gpx', help="Chemin du fichier GPX ou du dossier contenant les GPX")
    parser.add_argument('--all', action='store_true', help="Traite tous les fichiers GPX d'un dossier (power et/ou speed selon le mode choisi)")
    parser.add_argument('--cyclist', type=float, help="Poids du cycliste (kg)")
    parser.add_argument('--bike', type=float, default=9.0, help="Poids du velo (kg)")
    parser.add_argument('--accessories', type=float, default=3.0, help="Poids des accessoires (kg)")
    parser.add_argument('--power', type=float, help="Puissance moyenne (W) — requis pour le mode 'speed'")
    parser.add_argument('--group-distance', type=float, default=100.0, help="Distance cible (m) pour regrouper les segments")
    parser.add_argument('--min-slope', type=float, default=-0.5, help="Pente minimale (%%) pour inclure un groupe dans la moyenne")
    parser.add_argument('--chainrings', nargs='+', type=int, default=[50, 34], help="Liste des plateaux disponibles")
    parser.add_argument('--cogs', nargs='+', type=int, default=[13, 14, 15, 17, 19, 21, 23, 25, 27, 29], help="Liste des pignons disponibles")
    parser.add_argument('--rpm-range', nargs=2, type=float, default=[60.0, 80.0], help="Plage de cadence desiree (rpm min max)")
    parser.add_argument('--max-cog', type=int, default=None, help="Pignon maximum autorise avec le plus grand plateau")
    parser.add_argument('--scx', type=float, default=0.40, help="Coefficient aerodynamique SCx (defaut 0.40)")
    parser.add_argument('--mva', type=float, default=1.2, help="Masse volumique de l'air (defaut 1.2)")
    parser.add_argument('--kf', type=float, default=0.01, help="Coefficient de frottement Kf (defaut 0.01)")

    args = parser.parse_known_args()[0]
    args.chainrings = args.chainrings or [50, 34]
    args.cogs = args.cogs or [11, 12, 13, 14, 15, 17, 19, 21, 23, 26]
    args.rpm_range = tuple(args.rpm_range) if args.rpm_range else (60.0, 80.0)
    args.max_cog = args.max_cog

    if args.all:
        gpx_files = iter_gpx_files(args.gpx if args.gpx else '.')
        if not gpx_files:
            parser.error("Aucun fichier GPX trouve dans le dossier donne")

        modes = [args.mode] if args.mode else ['power', 'speed']
        for gpx_file in gpx_files:
            for selected_mode in modes:
                batch_args = argparse.Namespace(**vars(args))
                batch_args.gpx = gpx_file
                batch_args.mode = selected_mode
                batch_args.batch_mode = True
                if selected_mode == 'power':
                    mode_power(batch_args)
                else:
                    mode_speed(batch_args)
        return

    if not args.mode:
        parser.print_help()
        return

    if args.mode == 'speed' and args.power is None:
        parser.error("le mode 'speed' necessite --power")

    if args.mode == 'power':
        mode_power(args)
    else:
        mode_speed(args)


if __name__ == '__main__':
    main()
