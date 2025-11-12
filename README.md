# iwdd

## Requirements

- **Python 3.11+**
- **uv package manager** - https://docs.astral.sh/uv/getting-started/installation/

## Quick Start

1. **Setup:**

   ```bash
   # Clone the repository
   git clone https://github.com/00200200/iwdd.git
   cd iwdd

   # Install dependencies
   uv sync

   clone dataset from Google Drive to data/raw
   it should look like this:

   iwdd/
   └── data/
       └── raw/
            │──labels/
            └──videos/


   uv run -m src.scripts.train

   tensorboard --logdir lightning_logs/  Open http://localhost:6006 in your browser to view metrics.




   ```
