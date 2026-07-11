def compute_face_ratio(face_bbox: tuple, person_box: tuple) -> float:
    fx1, fy1, fx2, fy2 = face_bbox
    face_area = max(0, (fx2 - fx1) * (fy2 - fy1))

    px1, py1, px2, py2 = person_box
    person_area = max(0, (px2 - px1) * (py2 - py1))

    if person_area <= 0:
        return 0.0

    return face_area / person_area
