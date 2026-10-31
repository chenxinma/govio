/**
 * Govio 工具插件 — 将 govio-cli 包装为 DSH tools
 * 所有操作通过 govio-cli 子进程完成（argv 数组，无 shell），符合 skills 全局约束
 */
import { spawn } from 'node:child_process';
import { defineTool } from '@deepseek-ai/dsh-tools';
export const name = 'govio-tools';
export const inject = ['tools'];
/** govio-cli 允许转发的子命令白名单 */
const SUBCOMMANDS = new Set(['meta', 'query', 'observe', 'sql', 'backend']);
/** 默认子进程超时（毫秒） */
const DEFAULT_TIMEOUT_MS = 120_000;
/** 错误信息中 stderr 摘要的长度上限 */
const STDERR_EXCERPT_LIMIT = 2000;
/** 运行 govio-cli 子进程并收集 UTF-8 输出 */
async function runGovioCli(cli, args, signal, timeoutMs) {
    return new Promise((resolve, reject) => {
        const child = spawn(cli, args, { shell: false, stdio: ['ignore', 'pipe', 'pipe'] });
        let stdout = '';
        let stderr = '';
        let timedOut = false;
        let aborted = false;
        const timer = setTimeout(() => {
            timedOut = true;
            child.kill();
        }, timeoutMs);
        const onAbort = () => {
            aborted = true;
            child.kill();
        };
        if (signal.aborted) {
            onAbort();
        }
        else {
            signal.addEventListener('abort', onAbort, { once: true });
        }
        child.stdout.setEncoding('utf8');
        child.stderr.setEncoding('utf8');
        child.stdout.on('data', (chunk) => {
            stdout += chunk;
        });
        child.stderr.on('data', (chunk) => {
            stderr += chunk;
        });
        child.on('error', (error) => {
            clearTimeout(timer);
            signal.removeEventListener('abort', onAbort);
            reject(error);
        });
        child.on('close', (code) => {
            clearTimeout(timer);
            signal.removeEventListener('abort', onAbort);
            resolve({ code, stdout, stderr, timedOut, aborted });
        });
    });
}
export function apply(ctx, config) {
    const cli = config.cliPath ?? 'govio-cli';
    const timeoutMs = config.timeoutMs ?? DEFAULT_TIMEOUT_MS;
    /** 运行 govio-cli 并在失败时抛出中文错误（DSH 将抛错渲染为 isError 结果） */
    async function runOrFail(args, signal) {
        let result;
        try {
            result = await runGovioCli(cli, args, signal, timeoutMs);
        }
        catch (error) {
            const reason = error instanceof Error ? error.message : String(error);
            throw new Error(`无法启动 ${cli}（${reason}）；请确认已安装 govio-cli（uv tool install govio）并在 PATH 中，` +
                '或通过插件 config.cliPath 指定可执行文件路径');
        }
        if (result.aborted) {
            throw new Error('govio-cli 调用已取消');
        }
        if (result.timedOut) {
            throw new Error(`govio-cli 执行超时（${timeoutMs} ms）: ${args.join(' ')}`);
        }
        if (result.code !== 0) {
            throw new Error(`govio-cli 退出码 ${result.code}: ${result.stderr.slice(0, STDERR_EXCERPT_LIMIT)}`);
        }
        return result.stdout;
    }
    ctx.tools.register(defineTool({
        name: 'govio_cli',
        description: '执行 govio-cli 数据治理命令（子命令限 meta/query/observe/sql/backend）。' +
            '用于知识图谱维护、元数据/指标查询、数据探查与指标 SQL 组装。',
        parameters: {
            args: {
                type: 'array',
                items: { type: 'string' },
                required: true,
                description: "govio-cli 参数数组，如 ['query','-c','MATCH ...']",
            },
        },
        output: {
            schema: { type: 'string' },
            render: (_args, value) => [{ type: 'text', text: value }],
        },
        async execute({ args }, exec) {
            const [sub, ...rest] = args;
            if (!sub || !SUBCOMMANDS.has(sub)) {
                throw new Error(`不允许的子命令: ${sub ?? '(空)'}；允许: ${[...SUBCOMMANDS].join(', ')}`);
            }
            return runOrFail([sub, ...rest], exec.signal);
        },
    }));
    ctx.tools.register(defineTool({
        name: 'govio_query',
        description: '查询 govio 知识图谱（Cypher 或 Python，后端由 govio-cli 自动适配）',
        parameters: {
            query: {
                type: 'string',
                required: true,
                description: 'Cypher（以 MATCH 开头）或 NetworkX Python 代码',
            },
            language: {
                type: 'string',
                enum: ['cypher', 'python'],
                description: '查询语言提示，当前仅作参数校验，执行按图后端自适应',
            },
        },
        output: {
            schema: { type: 'string' },
            render: (_args, value) => [{ type: 'text', text: value }],
        },
        async execute({ query }, exec) {
            // language 现阶段仅作校验提示，govio-cli query 按图后端自适应解析
            return runOrFail(['query', '-c', query], exec.signal);
        },
    }));
}
