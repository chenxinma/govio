/**
 * Govio 工具扩展 — 将 govio-cli 包装为 Pi custom tools
 * 所有操作通过 govio-cli 子进程完成（argv 数组，无 shell），符合 skills 全局约束
 */
import { spawn } from "node:child_process";

import { Type } from "@earendil-works/pi-ai";
import { defineTool, type ExtensionAPI } from "@earendil-works/pi-coding-agent";

const CLI = process.env.GOVIO_CLI ?? "govio-cli";
const SUBCOMMANDS = new Set(["meta", "query", "observe", "sql", "backend"]);
const DEFAULT_TIMEOUT_MS = 120_000;
const DIAGNOSTIC_TIMEOUT_MS = 15_000;

/** govio-cli 子进程运行结果 */
interface CliResult {
	code: number | null;
	stdout: string;
	stderr: string;
}

/** 运行 govio-cli 子进程并收集 UTF-8 输出 */
async function runGovioCli(
	args: string[],
	signal: AbortSignal,
	timeoutMs = DEFAULT_TIMEOUT_MS,
): Promise<CliResult> {
	return await new Promise((resolve) => {
		const child = spawn(CLI, args, { shell: false, stdio: ["ignore", "pipe", "pipe"] });
		let stdout = "";
		let stderr = "";
		let timedOut = false;
		let aborted = false;

		// 超时与取消信号统一转发为 kill，由 close 事件收尾
		const timer = setTimeout(() => {
			timedOut = true;
			child.kill();
		}, timeoutMs);
		const onAbort = () => {
			aborted = true;
			child.kill();
		};
		signal.addEventListener("abort", onAbort, { once: true });

		child.stdout.setEncoding("utf8");
		child.stderr.setEncoding("utf8");
		child.stdout.on("data", (chunk: string) => {
			stdout += chunk;
		});
		child.stderr.on("data", (chunk: string) => {
			stderr += chunk;
		});

		// spawn 失败（如 govio-cli 不在 PATH）也按结果返回，由调用方统一转成错误
		child.on("error", (err: Error) => {
			clearTimeout(timer);
			signal.removeEventListener("abort", onAbort);
			resolve({ code: null, stdout, stderr: stderr || `无法启动 ${CLI}: ${err.message}` });
		});

		child.on("close", (code) => {
			clearTimeout(timer);
			signal.removeEventListener("abort", onAbort);
			if (timedOut) {
				resolve({ code: null, stdout, stderr: `${stderr}（执行超过 ${timeoutMs} ms，进程已终止）` });
				return;
			}
			if (aborted) {
				resolve({ code: null, stdout, stderr: `${stderr}（执行已取消）` });
				return;
			}
			resolve({ code, stdout, stderr });
		});
	});
}

/** 将非零退出/超时/取消/无法启动统一转为工具错误 */
function toToolError(result: CliResult): Error {
	const codeText = result.code === null ? "（超时/取消/无法启动）" : String(result.code);
	return new Error(`govio-cli 退出码 ${codeText}: ${result.stderr.slice(0, 2000) || result.stdout.slice(0, 2000)}`);
}

const govioCliTool = defineTool({
	name: "govio_cli",
	label: "Govio CLI",
	description:
		"执行 govio-cli 数据治理命令（子命令限 meta/query/observe/sql/backend）。" +
		"用于知识图谱维护、元数据/指标查询、数据探查与指标 SQL 组装。",
	parameters: Type.Object({
		args: Type.Array(Type.String(), { description: "govio-cli 参数数组，如 ['query','-c','MATCH ...']" }),
	}),
	async execute(_toolCallId, params, signal, _onUpdate, _ctx) {
		const [sub, ...rest] = params.args;
		if (!sub || !SUBCOMMANDS.has(sub)) {
			throw new Error(`不允许的子命令: ${sub ?? "(空)"}；允许: ${[...SUBCOMMANDS].join(", ")}`);
		}
		const r = await runGovioCli([sub, ...rest], signal);
		if (r.code !== 0) {
			throw toToolError(r);
		}
		return {
			content: [{ type: "text", text: r.stdout }],
			details: { args: params.args, code: r.code },
		};
	},
});

const govioQueryTool = defineTool({
	name: "govio_query",
	label: "Govio Query",
	description: "查询 govio 知识图谱（Cypher 或 Python，后端由 govio-cli 自动适配）",
	parameters: Type.Object({
		query: Type.String({ description: "Cypher（以 MATCH 开头）或 NetworkX Python 代码" }),
		language: Type.Optional(
			Type.Union([Type.Literal("cypher"), Type.Literal("python")], {
				description: "语法提示用；实际执行语法由图后端决定（NetworkX=Python，FalkorDB/Ladybug=Cypher）",
			}),
		),
	}),
	async execute(_toolCallId, params, signal, _onUpdate, _ctx) {
		if (!params.query.trim()) {
			throw new Error("query 不能为空");
		}
		const r = await runGovioCli(["query", "-c", params.query], signal);
		if (r.code !== 0) {
			throw toToolError(r);
		}
		return {
			content: [{ type: "text", text: r.stdout }],
			details: { query: params.query, language: params.language ?? "auto" },
		};
	},
});

export default function (pi: ExtensionAPI) {
	pi.registerTool(govioCliTool);
	pi.registerTool(govioQueryTool);
	pi.registerCommand("govio", {
		description: "Govio 环境自检（govio-cli 版本 / 后端 / assets 路径提示）",
		handler: async (_name, ctx) => {
			const signal = ctx.signal ?? new AbortController().signal;
			const versionResult = await runGovioCli(["-V"], signal, DIAGNOSTIC_TIMEOUT_MS);
			if (versionResult.code !== 0) {
				ctx.ui.notify(
					`govio-cli 不可用：${versionResult.stderr.trim() || "未知错误"}（请先 uv tool install govio）`,
					"error",
				);
				return;
			}
			const backendResult = await runGovioCli(["backend"], signal, DIAGNOSTIC_TIMEOUT_MS);
			const backendText =
				backendResult.code === 0 ? backendResult.stdout.trim() : `获取失败：${backendResult.stderr.trim()}`;
			ctx.ui.notify(
				[
					versionResult.stdout.trim(),
					`后端: ${backendText}`,
					"assets 约定位置: <skills-root>/govio/assets（由 govio-cli meta graph 生成）",
				].join("\n"),
				"info",
			);
		},
	});
}
