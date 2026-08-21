import cv2

def test_cameras():
    print("Testing camera indices...")
    for i in range(5):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            print(f"Camera index {i} is AVAILABLE.")
            cap.release()
        else:
            print(f"Camera index {i} is NOT available.")

if __name__ == "__main__":
    test_cameras()
