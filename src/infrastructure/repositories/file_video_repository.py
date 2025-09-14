import os
import json
from pathlib import Path
from typing import List, Optional, Dict
from src.application.interfaces.video_repository_interface import IVideoRepository


class FileVideoRepository(IVideoRepository):
    def __init__(self, input_path: str = "data/trailers"):
        self.input_path = Path(input_path)
        self._book_trailer_mapping = None  # Cache for mapping
        self._mapping_file_path = Path("data/book_trailer_mapping.json")

    def load_input_video(self, filename: str) -> Path:
        """Load input video from input_videos directory"""

        input_videos_path = Path("data/input_videos")
        full = input_videos_path / filename
        if not full.exists():
            raise FileNotFoundError(f"Video not found: {full}")
        return full

    def list_trailers(self) -> List[Path]:
        """List all available trailer files"""
        return [
            self.input_path / f
            for f in os.listdir(self.input_path)
            if f.lower().endswith((".mp4", ".avi", ".mov"))
        ]

    def get_trailer_for_book(self, book_name: str) -> Optional[str]:
        """
        Get trailer path for a specific book name based on JSON mapping

        Args:
            book_name: Name of the book (without extension)

        Returns:
            Path to trailer file or None if not found
        """
        # Load mapping if not cached
        if self._book_trailer_mapping is None:
            self._book_trailer_mapping = self._load_book_trailer_mapping()

        # Check direct mapping first
        if book_name in self._book_trailer_mapping:
            trailer_filename = self._book_trailer_mapping[book_name]
            trailer_path = self.input_path / trailer_filename

            if trailer_path.exists():
                return str(trailer_path)
            else:
                print(f"⚠️ Mapped trailer not found: {trailer_path}")

        # If it doesn't have a direct mapping, try to find a similar file
        print(f"🔍 Searching for similar trailer name for book: {book_name}")
        for trailer_file in self.input_path.glob("*.mp4"):
            if book_name.lower() in trailer_file.stem.lower():
                print(f"✅ Found similar trailer: {trailer_file}")
                return str(trailer_file)

        # Add the first available trailer for testing
        available_trailers = list(self.input_path.glob("*.mp4"))
        if available_trailers:
            print(f"📹 Using first available trailer for {book_name}: {available_trailers[0]}")
            return str(available_trailers)

        print(f"❌ No trailer found for book: {book_name}")
        return None

    def _load_book_trailer_mapping(self) -> Dict[str, str]:
        """
        Load book-to-trailer mapping from JSON file

        Returns:
            Dictionary mapping book names to trailer filenames
        """
        try:
            if self._mapping_file_path.exists():
                with open(self._mapping_file_path, 'r', encoding='utf-8') as f:
                    mapping = json.load(f)
                print(f"✅ Loaded book-trailer mapping from {self._mapping_file_path}")
                return mapping
            else:
                print(f"⚠️ Mapping file not found: {self._mapping_file_path}")
                print("📝 Creating default mapping file...")
                self._create_default_mapping_file()
                return {}
        except json.JSONDecodeError as e:
            print(f"❌ Error reading JSON mapping file: {e}")
            return {}
        except Exception as e:
            print(f"❌ Error loading book-trailer mapping: {e}")
            return {}

    def _create_default_mapping_file(self):
        """Create a default mapping file with sample mappings"""
        default_mapping = {
            {
                "A_Game_Of_Thrones_book": "Game of Thrones.mp4",
                "Angels_And_Demons_book": "Angels & Demons.mp4",
                "Da_Vinci_Code_book": "The Da Vinci Code.mp4",
                "Dracula_book": "Dracula.mp4",
                "Inferno_book": "Inferno.mp4",
                "New_Moon_book": "New Moon.mp4",
                "The_Girl_With_Dragon_Tattoo_book": "The Girl with the Dragon Tattoo.mp4",
                "The_Hobbit_book": "The Hobbit.mp4",
                "The_Lord_Of_The_Rings_Fellowship_book": "The Lord of the Rings- The Fellowship of the Ring - Official Trailer.mp4",
                "The_Lord_Of_The_Rings_Return_book": "The Lord of the Rings- The Return of the King - Official Trailer.mp4",
                "The_Lord_Of_The_Rings_Towers_book": "The Lord of the Rings- The Two Towers - Official Trailer.mp4",
                "Twilight_book": "Twilight.mp4",
            }

        }

        try:
            # Ensure data directory exists
            self._mapping_file_path.parent.mkdir(parents=True, exist_ok=True)

            with open(self._mapping_file_path, 'w', encoding='utf-8') as f:
                json.dump(default_mapping, f, indent=4, ensure_ascii=False)

            print(f"✅ Created default mapping file: {self._mapping_file_path}")

        except Exception as e:
            print(f"❌ Error creating default mapping file: {e}")

    def add_book_trailer_mapping(self, book_name: str, trailer_filename: str) -> bool:
        """
        Add or update a book-to-trailer mapping

        Args:
            book_name: Name of the book
            trailer_filename: Filename of the trailer

        Returns:
            True if successful, False otherwise
        """
        try:
            # Load current mapping
            if self._book_trailer_mapping is None:
                self._book_trailer_mapping = self._load_book_trailer_mapping()

            # Update mapping
            self._book_trailer_mapping[book_name] = trailer_filename

            # Save to file
            with open(self._mapping_file_path, 'w', encoding='utf-8') as f:
                json.dump(self._book_trailer_mapping, f, indent=4, ensure_ascii=False)

            print(f"✅ Added mapping: {book_name} -> {trailer_filename}")
            return True

        except Exception as e:
            print(f"❌ Error adding book-trailer mapping: {e}")
            return False

    def get_all_mappings(self) -> Dict[str, str]:
        """Get all book-to-trailer mappings"""
        if self._book_trailer_mapping is None:
            self._book_trailer_mapping = self._load_book_trailer_mapping()
        return self._book_trailer_mapping.copy()

    def reload_mapping(self):
        """Force reload the mapping from file"""
        self._book_trailer_mapping = None
        self._book_trailer_mapping = self._load_book_trailer_mapping()
