/**
 * Type definitions for the LLM chat application.
 */

export interface Env {
	/**
	 * Binding for the Workers AI API.
	 */
	AI: Ai;

	/**
	 * Binding for static assets.
	 */
	ASSETS: { fetch: (request: Request) => Promise<Response> };
}

/**
 * Represents a chat message.
 */
export interface ChatMessage {
	role: "system" | "user" | "assistant";
	content: string;
}

export type OpenAIContentPart = {
	type: "text";
	text: string;
};

export interface OpenAIChatMessage {
	role: "system" | "user" | "assistant" | "developer" | "tool";
	content: string | OpenAIContentPart[];
}

export interface OpenAIChatCompletionRequest {
	model?: string;
	messages: OpenAIChatMessage[];
	stream?: boolean;
	max_tokens?: number;
	temperature?: number;
	top_p?: number;
	presence_penalty?: number;
	frequency_penalty?: number;
	stop?: string | string[];
}
