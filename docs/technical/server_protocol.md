# Server Protocol Documentation

## 1. Overview

The system is built around a WebSocket connection. The Python client (written in client.py) streams raw audio from the microphone to the server, and the server (in server.py) processes the audio using voice activity detection (VAD) and automatic speech recognition (ASR). In addition, the client may send text-to-speech (TTS) requests and receive TTS audio back from the server. Two types of messages are exchanged over the WebSocket:

* Text messages (plain UTF‐8 strings) used for:
  * Authentication handshakes and control signals
  * Transcripts produced by ASR
  * TTS error messages
* Binary messages that follow a custom framing protocol ("MIRA" frames) used for:
  * TTS audio chunks
  * VAD (voice activity detection) status updates

## 2. Connection and Handshake

### Client Connection URI

When the client starts, it builds a connection URI using parameters read from a configuration file (config.json). The URI is of the form:

```
ws://<SERVER_HOST>:<SERVER_PORT>?api_key=<API_KEY>&client_id=<CLIENT_ID>&language=<LANGUAGE_CODE>
```

* `api_key` and `client_id` are provided as query parameters
* The client generates a unique client identifier (`client_id`) using a UUID
* `language` specifies the ISO language code (e.g., "en", "es", "fr") for TTS output

### Authentication

* Upon connection:
  * The server extracts the `api_key`, `client_id`, and `language` from the WebSocket's request URI
  * The server compares the provided `api_key` with its configured value and verifies that a client ID is present
  * The server stores the language code for the client session to use for TTS output
* Server Response:
  * If the API key (and client ID) is valid, the server sends the string `"AUTH_OK"`
  * If authentication fails, the server may send `"AUTH_FAILED"` (or close the connection)

> Note: All subsequent communication (audio and control messages) happens only after successful authentication.

## 3. Client-to-Server Messages

There are two main types of messages that the client sends:

### A. Audio Data

* What is sent:
  * The client continuously captures audio from the microphone, processes it (including converting stereo to mono, resampling to 16 kHz if necessary, and scaling to 16-bit integer samples), and sends the resulting raw PCM bytes.

* Format details:
  * Sample Format: 16-bit signed integer (little-endian)
  * Sampling Rate: The audio is resampled to 16 kHz (if the capture device rate differs)
  * Framing:
    * No additional header or framing is added by the client
    * The raw PCM bytes are sent as-is in each WebSocket binary message

* Purpose:
  * The server receives these binary messages and feeds them into its VAD/ASR pipeline (see Section 4 below)

### B. Control Messages (Text)

The client may also send plain text commands (UTF‐8 strings) that control server behavior. Recognized commands include:

#### TTS Requests:
* Format: A string starting with "TTS:" followed by the text to be converted to speech
* Example:
  ```
  TTS:Hello, how are you?
  ```
* Behavior:
  * The server accumulates the text until it has one or more complete sentences (ending with ., !, or ?)
  * Then processes it using its TTS pipeline
  * The resulting audio is sent back as binary frames (see Section 4B)

#### Other Control Commands:
* `"VOICE_FILTER_ON"`: Tells the server to enable any voice filtering (Not yet implemented, future feature)
* `"VOICE_FILTER_OFF"`: Tells the server to disable voice filtering
* `"RESET"`: Instructs the server to reset its internal audio buffers
* `"EXIT"`: Requests that the connection be closed

> Note: In the client code, any text entered via the GUI (if available) may be processed by an LLM (large language model) and then sent as a TTS request by prepending "TTS:" to the generated text.

## 4. Server-to-Client Messages

The server sends responses to the client in two distinct formats: plain text messages and binary messages that follow a custom framing protocol.

### A. Plain Text Messages

#### 1. Authentication and Transcript

* Authentication Response:
  * After the client connects and the server verifies the API key, the server sends:
    ```
    "AUTH_OK"
    ```
    (Or `"AUTH_FAILED"` if not – after which the connection is closed)

* ASR Transcripts:
  * When the server detects a complete utterance (using its VAD and ASR pipelines), it transcribes the audio
  * If the transcript passes filtering (for example, minimum length, confidence, and duplicate checks), the server sends the transcript as a plain text string
  * Example Transcript:
    ```
    Hello, this is a test transcript.
    ```

#### Error Messages:

If the TTS generation fails, the server sends the literal text:
```
TTS_ERROR
```

### B. Binary Messages (MIRA Framing Protocol)

The server uses a custom framing protocol (referred to as "MIRA") for sending binary data. This is used both for:

* TTS Audio Data: The TTS audio generated from text requests
* VAD Status Updates: Brief status updates indicating whether speech is currently being detected

#### Frame Structure

Each binary frame has the following layout:

* Magic Bytes (4 bytes):
  * Fixed value: `b'MIRA'`

* Frame Type (1 byte):
  * Indicates the kind of data in the frame:
    * `0x01`: Audio Data (Used for TTS audio chunks)
    * `0x02`: End of Utterance (Marks the end of a TTS audio sequence)
    * `0x03`: VAD Status (Indicates whether speech is currently detected; payload is a single byte)

* Payload Length (4 bytes):
  * A big-endian 32‐bit unsigned integer that specifies the length (in bytes) of the payload

* Payload (variable length):
  * For Frame Type `0x01` (Audio Data): Contains a chunk of PCM audio data (int16 samples)
  * For Frame Type `0x02` (End of Utterance): No payload (or empty)
  * For Frame Type `0x03` (VAD Status): A single byte:
    * `1` indicates that speech is currently detected
    * `0` indicates that no speech is detected

#### How Frames Are Used

##### 1. TTS Audio Transmission

When the server processes a TTS request:

* It generates TTS audio (typically as a NumPy array of floating‐point samples that are then scaled and converted to int16)
* The audio is segmented into chunks (typically with a fixed frame size, e.g. 512 samples per frame or other chunk sizes as determined by the implementation)
* For each chunk:
  * A frame is created with Frame Type `0x01` and the payload set to the int16 bytes of that chunk
  * These frames are sent to the client
* After all audio chunks have been sent:
  * A final frame with Frame Type `0x02` (End of Utterance) is sent to signal completion

##### 2. VAD Status Updates

* As the server processes the client's microphone audio (used for ASR), it periodically sends a VAD status frame:
  * The frame is built with Frame Type `0x03`
  * The payload is a single byte (`0x01` if speech is detected; `0x00` if not)
* The client uses these VAD frames (if running in GUI mode) to update visual indicators (for example, to show whether the user is speaking)

## 5. Detailed Message Flow Examples

### A. Connection and Authentication

1. Client initiates connection:
   * Request URI:
     ```
     ws://<SERVER_HOST>:<SERVER_PORT>?api_key=YOUR_API_KEY&client_id=some-uuid&language=en
     ```
2. Server verifies query parameters
3. Server sends:
   * Text message: `"AUTH_OK"`

### B. Streaming Audio from Client

1. Client captures audio, processes it (converts to mono, resamples to 16 kHz, converts to int16)

2. Client sends:
   * Binary messages containing raw PCM bytes

3. Server receives binary audio data:
   * Passes data to its `ServerAudioCore.process_audio()` method
   * The VAD (using Silero) runs over the incoming data
   * When a complete utterance is detected, the server runs ASR

4. Server sends transcript (if any):
   * Text message with the transcript string (e.g., `"Hello, how are you?"`)

5. During streaming, for VAD feedback:
   * Server sends VAD status frames:
     * Binary frame with header (magic MIRA, type `0x03`, payload length 1) and payload `[0x01]` (if speech is active) or `[0x00]`

### C. TTS Request/Response

1. Client sends a TTS request:
   * Text message starting with "TTS:" (e.g., `"TTS:Please read this text aloud."`)

2. Server processes the text:
   * The text is buffered until a sentence is complete
   * The TTS pipeline for the client's language is loaded or retrieved from cache
   * The TTS pipeline (using the Kokoro model) is invoked with the appropriate language settings

3. Server sends TTS audio:
   * A series of binary frames with Frame Type `0x01` are sent, each containing a chunk of int16 PCM TTS audio
   * After the audio is completely sent, a binary frame with Frame Type `0x02` is sent to mark the end of the utterance

4. In case of errors during TTS generation:
   * The server sends the text message `"TTS_ERROR"`

## 6. Error Handling and Cleanup

* Connection errors or too many consecutive sending/receiving errors cause the client to attempt a reconnection
* Cleanup:
  * Both client and server ensure that audio streams and buffers are properly closed and cleaned up on termination
* API key/ID verification failures result in an immediate error message and the connection being closed

## 7. Summary of Key Points for Client Developers

### Connection Setup
* Use the WebSocket protocol to connect to the server using the proper query parameters (`api_key`, `client_id`, and `language`)

### Audio Data (Client → Server)
* Send continuous binary messages containing raw PCM data:
  * 16-bit signed integers (no custom header)
  * Preferably at 16 kHz (or resampled to 16 kHz before sending)

### Control Messages (Client → Server)
* Send plain text strings for commands such as TTS requests (`"TTS:..."`), voice filter toggles, reset, or exit

### Transcripts (Server → Client)
* Expect plain text messages for ASR transcripts and authentication responses

### TTS Audio & VAD Status (Server → Client)
* Implement parsing for binary messages using the "MIRA" frame format:
  * Read the first 9 bytes for the header
  * Use the magic bytes (`"MIRA"`) to validate the frame
  * Process the frame type:
    * `0x01`: Append the payload (audio data) to a playback buffer
    * `0x02`: Mark the end of the current TTS utterance
    * `0x03`: Update UI indicators for VAD status using the single-byte payload

### Framing Protocol
* When parsing binary data, be aware that multiple frames might be concatenated, and partial frames may be received
* A robust client must handle buffering and splitting the frames correctly

## 8. Conclusion

This documentation summarizes the client/server communications:

* The client sends raw PCM audio and control text messages
* The server processes audio (using VAD and ASR), responds with transcript text messages, and sends TTS audio as a series of binary frames framed using a custom "MIRA" protocol
* VAD status is also communicated via special binary frames

By following this documentation and matching the message formats and protocols described here, a developer should be able to build an alternative client (for example, in JavaScript for a web browser) that can authenticate, stream audio to the server, handle transcripts, and receive/play back TTS audio as intended.
