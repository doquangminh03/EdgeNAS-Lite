from ultralytics import YOLO


def main() -> None:
    print("Loading YOLO26n...")

    model = YOLO("yolo26n.pt")

    results = model.predict(
        source="https://ultralytics.com/images/bus.jpg",
        device="cpu",
        save=True,
    )

    print("YOLO26 is working successfully.")
    print(f"Processed images: {len(results)}")
    print("Detection result saved inside runs/detect.")


if __name__ == "__main__":
    main()
    