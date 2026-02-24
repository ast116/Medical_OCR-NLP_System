def sort_boxes_top_to_bottom_left_to_right(results, y_threshold=10):
    """
    Trie les bounding boxes de haut en bas puis de gauche à droite
    """
    def sort_key(item):
        bbox = item[0]
        x_coords = [p[0] for p in bbox]
        y_coords = [p[1] for p in bbox]
        return (min(y_coords) // y_threshold, min(x_coords))

    return sorted(results, key=sort_key)


def group_boxes_by_line(results, overlap_ratio=0.5):
    """
    Regroupe les bounding boxes par ligne en utilisant
    le chevauchement vertical (robuste pour interlignes faibles)
    """

    lines = []

    for box, text, conf in results:
        y_coords = [p[1] for p in box]
        y_min, y_max = min(y_coords), max(y_coords)
        height = y_max - y_min

        assigned = False

        for line in lines:
            ly_min = line["y_min"]
            ly_max = line["y_max"]

            # Calcul du chevauchement vertical
            overlap = min(y_max, ly_max) - max(y_min, ly_min)

            if overlap > overlap_ratio * min(height, ly_max - ly_min):
                line["items"].append((box, text, conf))
                line["y_min"] = min(line["y_min"], y_min)
                line["y_max"] = max(line["y_max"], y_max)
                assigned = True
                break

        if not assigned:
            lines.append({
                "y_min": y_min,
                "y_max": y_max,
                "items": [(box, text, conf)]
            })

    # Tri horizontal interne
    for line in lines:
        line["items"].sort(
            key=lambda item: min(p[0] for p in item[0])
        )

    return lines


def reconstruct_text_from_lines(lines):
    """
    Reconstruit un texte ligne par ligne
    """
    reconstructed_lines = []

    for line in lines:
        texts = [item[1] for item in line["items"]]
        reconstructed_lines.append(" ".join(texts))

    return "\n".join(reconstructed_lines)


def is_suspicious_line(line, max_items=8):
    """
    Détecte une ligne anormalement chargée
    """
    return len(line["items"]) > max_items


def split_line_by_horizontal_gaps(line, gap_ratio=1.8):
    """
    Re-segmente une ligne en sous-lignes selon les écarts horizontaux
    """

    items = line["items"]
    items.sort(key=lambda item: min(p[0] for p in item[0]))

    xs = [min(p[0] for p in item[0]) for item in items]
    widths = [
        max(p[0] for p in item[0]) - min(p[0] for p in item[0])
        for item in items
    ]

    avg_width = sum(widths) / len(widths)
    new_lines = []
    current = [items[0]]

    for i in range(1, len(items)):
        gap = xs[i] - (xs[i-1] + widths[i-1])

        if gap > gap_ratio * avg_width:
            new_lines.append(current)
            current = [items[i]]
        else:
            current.append(items[i])

    new_lines.append(current)
    return new_lines


def refine_lines(lines):
    """
    Raffine les lignes OCR en re-segmentant uniquement
    les lignes suspectes (forte densité horizontale)
    """

    refined_lines = []

    for line in lines:
        if is_suspicious_line(line):
            split_groups = split_line_by_horizontal_gaps(line)

            for group in split_groups:
                refined_lines.append({
                    "items": group,
                    "y_min": min(p[1] for item in group for p in item[0]),
                    "y_max": max(p[1] for item in group for p in item[0]),
                })
        else:
            refined_lines.append(line)

    return refined_lines