import argparse
from ultralytics import YOLO

def validate_model(weights, data_yaml, imgsz):
    print(f"Validating model {weights} on dataset {data_yaml}")
    model = YOLO(weights)
    
    try:
        metrics = model.val(data=data_yaml, imgsz=imgsz)
        print("Validation completed.")
        return metrics
    except Exception as e:
        print(f"Validation failed: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--data", type=str, default="datasets/custom.yaml")
    parser.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args()
    
    validate_model(args.weights, args.data, args.imgsz)
