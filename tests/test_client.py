import requests
import argparse
import os

def main():
    parser = argparse.ArgumentParser(description="Test Cosmos API")
    parser.add_argument("--url", default="http://localhost:8000", help="API URL")
    parser.add_argument("--mode", choices=["video", "image"], required=True, help="Input mode")
    parser.add_argument("--input", required=True, help="Path to input file")
    parser.add_argument("--text", default="", help="Text prompt")
    parser.add_argument("--output", default="output.mp4", help="Output path")
    
    args = parser.parse_args()
    
    # Check if input file exists
    if not os.path.exists(args.input):
        print(f"Input file {args.input} does not exist")
        return
    
    # Prepare request
    files = {}
    data = {}
    
    if args.mode == "video":
        endpoint = f"{args.url}/api/v1/generate/from_video"
        files = {"video": open(args.input, "rb")}
    else:
        endpoint = f"{args.url}/api/v1/generate/from_image"
        files = {"image": open(args.input, "rb")}
    
    if args.text:
        data["text_prompt"] = args.text
    
    # Send request
    print(f"Sending request to {endpoint}")
    response = requests.post(endpoint, files=files, data=data)
    
    # Check response
    if response.status_code == 200:
        print("Request successful")
        with open(args.output, "wb") as f:
            f.write(response.content)
        print(f"Output saved to {args.output}")
    else:
        print(f"Request failed with status code {response.status_code}")
        print(response.text)

if __name__ == "__main__":
    main() 