# Language Integration with Kokoro Server

This document explains how to integrate the client's language-based prompting system with the Kokoro server's language support.

## Overview

The Mira Converse system supports multiple languages through two complementary mechanisms:

1. **Client-side language prompts**: Determines the language used for the LLM system prompt
2. **Server-side language settings**: Determines the language used for Text-to-Speech (TTS)

## Configuration Relationship

To provide a consistent user experience, you should align the client's prompt language with the server's TTS language:

| Client Config | Server Config |
|---------------|---------------|
| `llm.prompt.language` | `server.models.kokoro.language_code` |

## Language Codes

### Client Language Codes

The client uses standard language codes for prompt files:
- `en` - English
- `es` - Spanish
- `fr` - French
- etc.

### Server Language Codes

The Kokoro server uses single-letter language codes:
- `a` - Default (English)
- `e` - Spanish
- `f` - French
- etc.

## Configuration Example

### Client Configuration (client/default_config.json)

```json
"llm": {
  "prompt": {
    "language": "es",
    "custom_path": null,
    "directory": "prompts"
  }
}
```

### Server Configuration (server/default_config.json)

```json
"server": {
  "models": {
    "kokoro": {
      "path": "...",
      "voice_name": "af_heart",
      "language_code": "e"  // Spanish
    }
  }
}
```

## Environment Variables

You can also set these values using environment variables:

- Client language: Set in client configuration
- Server language: `KOKORO_LANGUAGE_CODE` environment variable

## Adding New Languages

When adding support for a new language:

1. Create a new prompt file in `client/prompts/default/` with the appropriate language code
2. Update the server configuration to use the corresponding Kokoro language code
3. Ensure the voice model supports the selected language

## Language Fallback

If a language is not supported by either component:

- Client will fall back to English prompts if the specified language is not available
- Server will use the default language code (`a`) if the specified language is not supported
