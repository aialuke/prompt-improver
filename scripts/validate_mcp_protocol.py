import json
import sys


def validate_mcp_protocol(filename):
    try:
        with open(filename, encoding="utf-8") as file:
            data = file.read()
            parsed = json.loads(data)
            if not isinstance(parsed, dict) or "methods" not in parsed:
                raise ValueError("Invalid MCP protocol structure")
            print(f"✅ {filename}: MCP protocol valid")
            return True
    except OSError as e:
        print(f"❌ {filename}: File I/O error - {e}")
        return False
    except json.JSONDecodeError as e:
        print(f"❌ {filename}: Invalid JSON format - {e}")
        return False
    except (ValueError, TypeError) as e:
        print(f"❌ {filename}: MCP protocol validation failed - {e}")
        return False
    except KeyboardInterrupt:
        print("\n⚠️ Validation cancelled by user")
        raise
    except Exception as e:
        print(f"❌ {filename}: Unexpected validation error - {e}")
        import logging

        logging.exception("Unexpected error validating %s", filename)
        return False


if __name__ == "__main__":
    filenames = sys.argv[1:]
    success = all(validate_mcp_protocol(filename) for filename in filenames)
    if not success:
        sys.exit(1)
