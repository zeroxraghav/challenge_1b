# Adobe Hackathon: Round 1B - Persona-Driven Document Intelligence

This project is a solution for Round 1B of the Adobe Hackathon, "Connecting the Dots." It provides an intelligent system that processes collections of PDF documents, analyzes their content, and ranks sections based on their relevance to a specific user persona and their job-to-be-done.

## How to Build and Run

The solution is containerized using Docker and is designed to process all `Collection X/` directories found in its root.

### Build the Docker Image

From the root directory (where the `Dockerfile` is located), run the following command:

```bash
docker build --platform linux/amd64 -t challenge1b .

Run the Solution
The following command will run the container. It mounts the current directory into the container's /app folder, allowing the script to find and process all collections. The script will automatically generate a challenge1b_output.json inside each collection folder.

docker run --rm -v "$(pwd):/app" --network none challenge1b
```
