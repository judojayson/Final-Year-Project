def predict_gait_from_video(video_path):
    import cv2
    import mediapipe as mp
    import numpy as np
    import joblib
    import tensorflow as tf

    scaler = joblib.load("gait_project_66/models/scaler.pkl")
    model = tf.keras.models.load_model("gait_project_66/models/gait_classifier.keras")

    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()
    cap = cv2.VideoCapture(video_path)

    features = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = pose.process(rgb)
        if result.pose_landmarks:
            row = []
            for lm in result.pose_landmarks.landmark[:33]:
                row.extend([lm.x, lm.y])
            features.append(row)

    cap.release()
    pose.close()

    if not features:
        return "No Pose Detected", 0.0

    X = scaler.transform(features)
    preds = model.predict(X)
    classes = np.argmax(preds, axis=1)

    majority = np.bincount(classes).argmax()
    label_map = {0: "Normal Gait", 1: "Cerebellar Gait"}
    confidence = np.bincount(classes)[majority] / len(classes)

    return label_map[majority], confidence * 100
