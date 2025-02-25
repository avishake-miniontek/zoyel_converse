# Mira Converse Prompt System

This directory contains the prompt templates used by Mira Converse for different languages and custom configurations.

## Directory Structure

```
prompts/
  ├── default/         # Default language-specific prompts
  │   ├── en.json      # English prompt (default)
  │   ├── es.json      # Spanish prompt
  │   └── ...          # Other language prompts
  └── custom/          # Custom user-defined prompts
      └── example_custom.json  # Example custom prompt
```

## Prompt Configuration

Prompts are configured in the client's `default_config.json` file under the `llm.prompt` section:

```json
"llm": {
  "prompt": {
    "language": "en",           // Language code for default prompts
    "custom_path": null,        // Optional path to custom prompt file
    "directory": "prompts"      // Directory containing prompt files
  }
}
```

### Configuration Options

- `language`: The language code to use for default prompts (e.g., "en", "es", "fr")
- `custom_path`: Optional path to a custom prompt file. If provided, this will override the language setting.
- `directory`: The directory containing prompt files (relative to the client directory)

## Prompt File Format

Prompt files are JSON files with the following structure:

```json
{
  "system_prompt": "You are {assistant_name}, a helpful AI assistant...",
  "language": "en",
  "language_name": "English"
}
```

### Required Fields

- `system_prompt`: The system prompt text to send to the LLM. Can include variables like `{assistant_name}`.
- `language`: The language code for this prompt.
- `language_name`: A human-readable name for the language.

## Creating Custom Prompts

To create a custom prompt:

1. Create a new JSON file with the required fields (see format above)
2. Place it in the `prompts/custom/` directory or any location of your choice
3. Update the `custom_path` in your config to point to your custom prompt file

Example custom prompt configuration:

```json
"llm": {
  "prompt": {
    "language": "en",
    "custom_path": "prompts/custom/my_custom_prompt.json",
    "directory": "prompts"
  }
}
```

## Variables in Prompts

The following variables can be used in prompts:

- `{assistant_name}`: Replaced with the assistant's name from the config

## Fallback Behavior

If a prompt cannot be loaded, the system follows this fallback sequence:

1. Try to load the custom prompt if specified
2. Try to load the language-specific prompt
3. Fall back to English if available
4. Use a hardcoded default prompt as a last resort

## Adding New Languages

To add support for a new language:

1. Create a new JSON file in the `prompts/default/` directory
2. Name it with the appropriate language code (e.g., `fr.json` for French)
3. Include all required fields in the proper format
4. Set the `language` config option to your new language code
