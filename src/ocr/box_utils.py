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


def group_boxes_by_line(results, y_threshold=15):
    """
    Regroupe les bounding boxes appartenant à la même ligne
    """
    lines = []

    for bbox, text, conf in results:
        y_min = min(p[1] for p in bbox)
        placed = False

        for line in lines:
            if abs(line["y"] - y_min) < y_threshold:
                line["items"].append((bbox, text, conf))
                placed = True
                break

        if not placed:
            lines.append({
                "y": y_min,
                "items": [(bbox, text, conf)]
            })

    # Trier les lignes de haut en bas
    lines.sort(key=lambda l: l["y"])

    # Trier les éléments de chaque ligne de gauche à droite
    for line in lines:
        line["items"].sort(key=lambda i: min(p[0] for p in i[0]))

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