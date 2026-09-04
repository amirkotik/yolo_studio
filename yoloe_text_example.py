import cv2
import time
import threading
from ultralytics import YOLOE

current_classes = ["watch"]
update_classes_flag = False

def input_thread():
    global current_classes, update_classes_flag
    while True:
        user_input = input("\n[ВВОД] Что ищем? (через запятую): ")
        if user_input.strip():
            current_classes = [item.strip() for item in user_input.split(",")]
            update_classes_flag = True

def main():
    global current_classes, update_classes_flag

    model = YOLOE("yoloe-26x-seg.pt")
    model.set_classes(current_classes)

    t = threading.Thread(target=input_thread, daemon=True)
    t.start()

    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    if not cap.isOpened():
        print("Ошибка: Не удалось открыть видеопоток.")
        return

    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    prev_frame_time = 0


    while True:
        success, frame = cap.read()
        if not success:
            break

        if update_classes_flag:
            print(f"\n[СИСТЕМА] Перестраиваю зрение на: {current_classes}")
            model.set_classes(current_classes)
            update_classes_flag = False

        results = model.predict(frame, conf=0.05, imgsz=256, verbose=False)
        annotated_frame = results[0].plot()

        new_frame_time = time.time()
        if (new_frame_time - prev_frame_time) > 0:
            fps = 1 / (new_frame_time - prev_frame_time)
        else:
            fps = 0
        prev_frame_time = new_frame_time

        cv2.putText(annotated_frame, f"FPS: {int(fps)}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Dynamic YOLOE", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()