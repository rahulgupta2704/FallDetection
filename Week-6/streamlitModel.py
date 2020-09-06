import streamlit as st 
import io
import tensorflow as tf
import cv2
import time
import argparse
import pandas as pd
import posenet
from tensorflow.keras.models import load_model
import numpy as np
from sklearn.preprocessing import StandardScaler

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=int, default=101)
parser.add_argument('--cam_id', type=int, default=0)
parser.add_argument('--cam_width', type=int, default=1280)
parser.add_argument('--cam_height', type=int, default=720)
parser.add_argument('--scale_factor', type=float, default=0.7125)
parser.add_argument('--file', type=str, default=None, help="Optionally use a video file instead of a live camera")
args = parser.parse_args()


def action():
    videoPlayer = st.empty()
    
    with tf.Session() as sess:
        model_cfg, model_outputs = posenet.load_model(args.model, sess)
        output_stride = model_cfg['output_stride']

        if args.file is not None:
            cap = cv2.VideoCapture(args.file)
        else:
            cap = cv2.VideoCapture("upload.mp4")
        cap.set(3, args.cam_width)
        cap.set(4, args.cam_height)

        start = time.time()
        frame_count = 0

        df = pd.DataFrame(columns=['nose_x', 'nose_y', 'leftEye_x', 'leftEye_y', 'rightEye_x',
       'rightEye_y', 'leftEar_x', 'leftEar_y', 'rightEar_x', 'rightEar_y',
       'leftShoulder_x', 'leftShoulder_y', 'rightShoulder_x',
       'rightShoulder_y', 'leftElbow_x', 'leftElbow_y', 'rightElbow_x',
       'rightElbow_y', 'leftWrist_x', 'leftWrist_y', 'rightWrist_x',
       'rightWrist_y', 'leftHip_x', 'leftHip_y', 'rightHip_x', 'rightHip_y',
       'leftKnee_x', 'leftKnee_y', 'rightKnee_x', 'rightKnee_y', 'leftAnkle_x',
       'leftAnkle_y', 'rightAnkle_x', 'rightAnkle_y'])
        row = 0
        feed_num = 0
        model = load_model('RNN2.h5')
        scaler = StandardScaler()
        result = 'Loading...'
        # font 
        font = cv2.FONT_HERSHEY_SIMPLEX 
          
        # org 
        org = (50, 50) 
          
        # fontScale 
        fontScale = 1
           
        # Blue color in BGR 
        color = (255, 0, 0) 
          
        # Line thickness of 2 px 
        thickness = 2

        while True:
            input_image, display_image, output_scale = posenet.read_cap(
                cap, scale_factor=args.scale_factor, output_stride=output_stride)

            heatmaps_result, offsets_result, displacement_fwd_result, displacement_bwd_result = sess.run(
                model_outputs,
                feed_dict={'image:0': input_image}
            )

            pose_scores, keypoint_scores, keypoint_coords = posenet.decode_multi.decode_multiple_poses(
                heatmaps_result.squeeze(axis=0),
                offsets_result.squeeze(axis=0),
                displacement_fwd_result.squeeze(axis=0),
                displacement_bwd_result.squeeze(axis=0),
                output_stride=output_stride,
                max_pose_detections=1,
                min_pose_score=0.15)

            keypoint_coords *= output_scale

            # TODO this isn't particularly fast, use GL for drawing and display someday...
            overlay_image = posenet.draw_skel_and_kp(
                display_image, pose_scores, keypoint_scores, keypoint_coords,
                min_pose_score=0.15, min_part_score=0.1)
            
            keypoints = {'nose_x': [keypoint_coords[0][0][0]], 'nose_y': [keypoint_coords[0][1][1]], 'leftEye_x': [keypoint_coords[0][1][0]], 'leftEye_y': [keypoint_coords[0][2][1]], 'rightEye_x': [keypoint_coords[0][3][0]], 'rightEye_y': [keypoint_coords[0][3][1]], 'leftEar_x': [keypoint_coords[0][4][0]], 'leftEar_y': [keypoint_coords[0][4][1]], 'rightEar_x': [keypoint_coords[0][0][0]], 'rightEar_y': [keypoint_coords[0][0][0]], 'leftShoulder_x': [keypoint_coords[0][5][0]], 'leftShoulder_y': [keypoint_coords[0][5][1]], 'rightShoulder_x': [keypoint_coords[0][6][0]], 'rightShoulder_y': [keypoint_coords[0][6][1]], 'leftElbow_x': [keypoint_coords[0][7][0]], 'leftElbow_y': [keypoint_coords[0][7][1]], 'rightElbow_x': [keypoint_coords[0][8][0]], 'rightElbow_y': [keypoint_coords[0][8][1]], 'leftWrist_x': [keypoint_coords[0][9][0]], 'leftWrist_y': [keypoint_coords[0][9][1]], 'rightWrist_x': [keypoint_coords[0][10][0]], 'rightWrist_y': [keypoint_coords[0][10][1]], 'leftHip_x': [keypoint_coords[0][11][0]], 'leftHip_y': [keypoint_coords[0][11][1]], 'rightHip_x': [keypoint_coords[0][12][0]], 'rightHip_y': [keypoint_coords[0][12][1]], 'leftKnee_x': [keypoint_coords[0][13][0]], 'leftKnee_y': [keypoint_coords[0][13][1]], 'rightKnee_x': [keypoint_coords[0][14][0]], 'rightKnee_y': [keypoint_coords[0][14][1]], 'leftAnkle_x': [keypoint_coords[0][15][0]], 'leftAnkle_y': [keypoint_coords[0][15][1]], 'rightAnkle_x': [keypoint_coords[0][16][0]], 'rightAnkle_y': [keypoint_coords[0][16][1]]}
            row =  pd.DataFrame(data=keypoints)
            df = df.append(row, ignore_index=True)
            if len(df) >= 120:
                if len(df) % 30 == 0:
                    x = df[feed_num:feed_num + 120]
                    x = scaler.fit_transform(x)
                    x = x.reshape(1,120,34)
                    y_pred = model.predict_classes(x)
                    if y_pred[0] == 0 :
                        result = 'falling'
                        print('falling')
                    elif y_pred[0] == 1 :
                        result = 'pushups'
                        print('pushups')
                    elif y_pred[0] == 2 :
                        result = 'sitting'
                        print('sitting')
                    elif y_pred[0] == 3 :
                        result = 'walking'
                        print('walking')
                feed_num += 1

            cv2.putText(overlay_image, 'Action: '+result, org, font, fontScale, color, thickness, cv2.LINE_AA, False)
            videoPlayer.image(overlay_image, channels="BGR")
            frame_count += 1
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                videoPlayer = st.empty()
                cap.release()
                cv2.destroyAllWindows()
                break
            if frame_count == int(cap.get(cv2.CAP_PROP_FRAME_COUNT)):
                videoPlayer = st.empty()
                cap.release()
                cv2.destroyAllWindows()
                break

        print('Average FPS: ', frame_count / (time.time() - start))
        
def main():
    st.set_option('deprecation.showfileUploaderEncoding', False)
    st.title("Fall Detection")
    st.text("Upload a video to detect action")

    video_file = st.file_uploader("Upload File...", type=["mp4"])
    temporary_location = False
    
    if video_file is not None:
        g = io.BytesIO(video_file.read())
        temporary_location = "upload.mp4"
        with open(temporary_location, 'wb') as out:
            out.write(g.read())
        out.close()
        action()


if __name__ == "__main__":
    main()