from time import time
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from multiprocessing import Pool


def DangerousDrivingDetection(frames, 
                                feature_extractor_path='feature_extractor.h5',
                                sequence_model_path='sequence_model.h5',
                                IMG_SIZE = 224,
                                MAX_SEQ_LENGTH = 20,
                                NUM_FEATURES = 576):

    feature_extractor = load_model(feature_extractor_path)
    sequence_model = load_model(sequence_model_path)
    IMG_SIZE = IMG_SIZE
    MAX_SEQ_LENGTH = MAX_SEQ_LENGTH
    NUM_FEATURES = NUM_FEATURES


    start = time()

    def load_video(frames):
        return np.array(frames)



    def prepare_single_video(frames):
        frames = frames[None, ...]
        frame_mask = np.zeros(shape=(1, MAX_SEQ_LENGTH,), dtype="bool")
        frame_features = np.zeros(shape=(1, MAX_SEQ_LENGTH, NUM_FEATURES), dtype="float32")

        for i, batch in enumerate(frames):
            video_length = batch.shape[0]
            length = min(MAX_SEQ_LENGTH, video_length)
            for j in range(length):
                frame_features[i, j, :] = feature_extractor.predict(batch[None, j, :])
            frame_mask[i, :length] = 1  # 1 = not masked, 0 = masked

        return frame_features, frame_mask


    def sequence_prediction(frames):
        
        class_vocab = ['Dangerous', 'Safe']
        frames = load_video(frames)
        start = time()
        frame_features, frame_mask = prepare_single_video(frames)
        probabilities = sequence_model.predict([frame_features, frame_mask])[0]
        end = time()
        print(f'Inference Time: {end - start}')
        inference_time = end - start

        for i in np.argsort(probabilities)[::-1]:
            print(f"{i}  {class_vocab[i]}: {probabilities[i] * 100:5.2f}%")
        
        if probabilities[0] > probabilities[1]:
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640,480))
            for frame in frames:
                out.write(frame)

        return inference_time, frames

    inference_time, test_frames = sequence_prediction(frames)

    end = time()

    duration = end - start
    print(duration)


if __name__ == '__main__':
    cap = cv2.VideoCapture("Dangerous_10.avi")

    counter = 0
    frames = []
    while True:

        ret, frame = cap.read()
        counter += 1
        IMG_SIZE = 224
        frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
        cv2.imshow("Camera", frame)
        frame = frame[:, :, [2, 1, 0]]
        frames.append(frame)

        if counter == 90:

            # DangerousDrivingDetection(frames=frames,
                                        # sequence_model_path = "lstm.h5")
            with Pool(8) as p:
                p.map(DangerousDrivingDetection, [frames])
            counter = 0
            frames = []
        else:
            pass

        # if cv2.waitKey(0) == ord("q"):
        #     break
        #     cap.release()
        #     cv2.destroyAllWindows()
        # else:
        #     pass

