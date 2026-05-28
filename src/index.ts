/**
 * LLM Chat Application Template
 *
 * A simple chat application using Cloudflare Workers AI.
 * This template demonstrates how to implement an LLM-powered chat interface with
 * streaming responses using Server-Sent Events (SSE).
 *
 * @license MIT
 */
import {
	ChatMessage,
	Env,
	OpenAIChatCompletionRequest,
	OpenAIChatMessage,
} from "./types";

// Model ID for Workers AI model
// https://developers.cloudflare.com/workers-ai/models/
const MODEL_ID = "@cf/meta/llama-3.1-8b-instruct-fp8";

// Default system prompt
const SYSTEM_PROMPT =
	"You are a helpful, friendly assistant. Provide concise and accurate responses.";

const OPENAI_COMPLETIONS_PATH = "/v1/chat/completions";
const OPENAI_OBJECT = "chat.completion";
const OPENAI_CHUNK_OBJECT = "chat.completion.chunk";

export default {
	/**
	 * Main request handler for the Worker
	 */
	async fetch(
		request: Request,
		env: Env,
		ctx: ExecutionContext,
	): Promise<Response> {
		const url = new URL(request.url);

		if (url.pathname === OPENAI_COMPLETIONS_PATH) {
			if (request.method === "POST") {
				return handleOpenAIChatRequest(request, env);
			}

			return openAIError("Method not allowed", 405);
		}

		// Handle static assets (frontend)
		if (url.pathname === "/" || !url.pathname.startsWith("/api/")) {
			return env.ASSETS.fetch(request);
		}

		// API Routes
		if (url.pathname === "/api/chat") {
			// Handle POST requests for chat
			if (request.method === "POST") {
				return handleChatRequest(request, env);
			}

			// Method not allowed for other request types
			return new Response("Method not allowed", { status: 405 });
		}

		// Handle 404 for unmatched routes
		return new Response("Not found", { status: 404 });
	},
} satisfies ExportedHandler<Env>;

/**
 * Handles chat API requests
 */
async function handleChatRequest(
	request: Request,
	env: Env,
): Promise<Response> {
	try {
		// Parse JSON request body
		const { messages = [] } = (await request.json()) as {
			messages: ChatMessage[];
		};

		// Add system prompt if not present
		if (!messages.some((msg) => msg.role === "system")) {
			messages.unshift({ role: "system", content: SYSTEM_PROMPT });
		}

		const stream = await env.AI.run(
			MODEL_ID,
			{
				messages,
				max_tokens: 1024,
				stream: true,
			},
			{
				// Uncomment to use AI Gateway
				// gateway: {
				//   id: "YOUR_GATEWAY_ID", // Replace with your AI Gateway ID
				//   skipCache: false,      // Set to true to bypass cache
				//   cacheTtl: 3600,        // Cache time-to-live in seconds
				// },
			},
		);

		return new Response(stream, {
			headers: {
				"content-type": "text/event-stream; charset=utf-8",
				"cache-control": "no-cache",
				connection: "keep-alive",
			},
		});
	} catch (error) {
		console.error("Error processing chat request:", error);
		return new Response(
			JSON.stringify({ error: "Failed to process request" }),
			{
				status: 500,
				headers: { "content-type": "application/json" },
			},
		);
	}
}

async function handleOpenAIChatRequest(
	request: Request,
	env: Env,
): Promise<Response> {
	let body: OpenAIChatCompletionRequest;
	try {
		body = (await request.json()) as OpenAIChatCompletionRequest;
	} catch (error) {
		console.error("Error parsing OpenAI request:", error);
		return openAIError("Invalid JSON in request body", 400);
	}

	if (!Array.isArray(body.messages) || body.messages.length === 0) {
		return openAIError("Request must include a non-empty messages array", 400);
	}

	const messages = normalizeOpenAIMessages(body.messages);
	if (!messages.length) {
		return openAIError("Messages must include text content", 400);
	}

	if (!messages.some((message) => message.role === "system")) {
		messages.unshift({ role: "system", content: SYSTEM_PROMPT });
	}

	const model = typeof body.model === "string" && body.model ? body.model : MODEL_ID;
	const aiOptions: Record<string, unknown> = {
		messages,
		max_tokens: body.max_tokens ?? 1024,
	};

	if (typeof body.temperature === "number") {
		aiOptions.temperature = body.temperature;
	}

	if (typeof body.top_p === "number") {
		aiOptions.top_p = body.top_p;
	}

	if (typeof body.presence_penalty === "number") {
		aiOptions.presence_penalty = body.presence_penalty;
	}

	if (typeof body.frequency_penalty === "number") {
		aiOptions.frequency_penalty = body.frequency_penalty;
	}

	if (body.stop) {
		aiOptions.stop = body.stop;
	}

	try {
		if (body.stream) {
			const stream = (await env.AI.run(
				model,
				{ ...aiOptions, stream: true },
				{},
			)) as ReadableStream<Uint8Array>;

			return new Response(
				createOpenAIStream(stream, model),
				openAIStreamHeaders(),
			);
		}

		const result = await env.AI.run(model, aiOptions, {});
		const responseText = extractResponseText(result);
		if (!responseText) {
			return openAIError("No response generated", 500);
		}

		const payload: Record<string, unknown> = {
			id: createOpenAIId(),
			object: OPENAI_OBJECT,
			created: Math.floor(Date.now() / 1000),
			model,
			choices: [
				{
					index: 0,
					message: { role: "assistant", content: responseText },
					finish_reason: "stop",
				},
			],
		};

		const usage = extractUsage(result);
		if (usage) {
			payload.usage = usage;
		}

		return new Response(JSON.stringify(payload), {
			headers: { "content-type": "application/json" },
		});
	} catch (error) {
		console.error("Error processing OpenAI request:", error);
		return openAIError("Failed to process request", 500);
	}
}

function normalizeOpenAIMessages(messages: OpenAIChatMessage[]): ChatMessage[] {
	return messages
		.map((message) => {
			const content = normalizeOpenAIContent(message.content);
			if (!content) {
				return null;
			}

			const role = normalizeOpenAIRole(message.role);
			if (!role) {
				return null;
			}

			return { role, content };
		})
		.filter((message): message is ChatMessage => Boolean(message));
}

function normalizeOpenAIRole(
	role: OpenAIChatMessage["role"],
): ChatMessage["role"] | null {
	if (role === "system" || role === "user" || role === "assistant") {
		return role;
	}

	if (role === "developer") {
		return "system";
	}

	return null;
}

function normalizeOpenAIContent(
	content: OpenAIChatMessage["content"],
): string | null {
	if (typeof content === "string") {
		return content;
	}

	if (!Array.isArray(content)) {
		return null;
	}

	const parts = content
		.map((part) => (part.type === "text" ? part.text : ""))
		.filter(Boolean);

	return parts.length ? parts.join("") : null;
}

function extractResponseText(result: unknown): string | null {
	if (typeof result === "string") {
		return result;
	}

	if (result && typeof result === "object" && "response" in result) {
		const response = (result as { response?: unknown }).response;
		if (typeof response === "string") {
			return response;
		}
	}

	return null;
}

function extractUsage(result: unknown): Record<string, number> | null {
	if (result && typeof result === "object" && "usage" in result) {
		const usage = (result as { usage?: unknown }).usage;
		if (
			usage &&
			typeof usage === "object" &&
			"total_tokens" in usage &&
			typeof (usage as { total_tokens: unknown }).total_tokens === "number"
		) {
			return usage as Record<string, number>;
		}
	}

	return null;
}

function createOpenAIStream(
	stream: ReadableStream<Uint8Array>,
	model: string,
): ReadableStream<Uint8Array> {
	const reader = stream.getReader();
	const decoder = new TextDecoder();
	const encoder = new TextEncoder();
	const id = createOpenAIId();
	const created = Math.floor(Date.now() / 1000);
	let buffer = "";
	let sentRole = false;
	let finished = false;

	const finalize = (controller: ReadableStreamDefaultController<Uint8Array>) => {
		if (finished) {
			return;
		}

		finished = true;
		const finishChunk = {
			id,
			object: OPENAI_CHUNK_OBJECT,
			created,
			model,
			choices: [
				{
					index: 0,
					delta: {},
					finish_reason: "stop",
				},
			],
		};
		controller.enqueue(
			encoder.encode(`data: ${JSON.stringify(finishChunk)}\n\n`),
		);
		controller.enqueue(encoder.encode("data: [DONE]\n\n"));
	};

	return new ReadableStream<Uint8Array>({
		async pull(controller) {
			const { value, done } = await reader.read();
			if (done) {
				finalize(controller);
				controller.close();
				return;
			}

			buffer += decoder.decode(value, { stream: true });
			let boundaryIndex = buffer.indexOf("\n\n");
			while (boundaryIndex !== -1) {
				const chunk = buffer.slice(0, boundaryIndex);
				buffer = buffer.slice(boundaryIndex + 2);
				boundaryIndex = buffer.indexOf("\n\n");

				const line = chunk
					.split("\n")
					.map((entry) => entry.trim())
					.find((entry) => entry.startsWith("data:"));
				if (!line) {
					continue;
				}

				const data = line.slice(5).trim();
				if (!data || data === "[DONE]") {
					continue;
				}

				let payload: unknown;
				try {
					payload = JSON.parse(data);
				} catch {
					continue;
				}

				const deltaText = extractStreamDelta(payload);
				const doneSignal = extractStreamDone(payload);
				if (deltaText) {
					const delta: Record<string, string> = { content: deltaText };
					if (!sentRole) {
						delta.role = "assistant";
						sentRole = true;
					}

					const openAIChunk = {
						id,
						object: OPENAI_CHUNK_OBJECT,
						created,
						model,
						choices: [
							{
								index: 0,
								delta,
								finish_reason: null,
							},
						],
					};
					controller.enqueue(
						encoder.encode(`data: ${JSON.stringify(openAIChunk)}\n\n`),
					);
				}

				if (doneSignal) {
					finalize(controller);
					controller.close();
					return;
				}
			}
		},
		cancel() {
			reader.cancel().catch(() => undefined);
		},
	});
}

function extractStreamDelta(payload: unknown): string | null {
	if (!payload || typeof payload !== "object") {
		return null;
	}

	if ("response" in payload && typeof payload.response === "string") {
		return payload.response;
	}

	if ("delta" in payload && typeof payload.delta === "string") {
		return payload.delta;
	}

	if ("text" in payload && typeof payload.text === "string") {
		return payload.text;
	}

	return null;
}

function extractStreamDone(payload: unknown): boolean {
	if (!payload || typeof payload !== "object") {
		return false;
	}

	if ("done" in payload && typeof payload.done === "boolean") {
		return payload.done;
	}

	return false;
}

function openAIStreamHeaders(): ResponseInit {
	return {
		headers: {
			"content-type": "text/event-stream; charset=utf-8",
			"cache-control": "no-cache",
			connection: "keep-alive",
		},
	};
}

function createOpenAIId(): string {
	return `chatcmpl-${crypto.randomUUID()}`;
}

function openAIError(message: string, status: number): Response {
	const payload = {
		error: {
			message,
			type: "invalid_request_error",
			param: null,
			code: null,
		},
	};
	return new Response(JSON.stringify(payload), {
		status,
		headers: { "content-type": "application/json" },
	});
}
