import type { Context } from '@deepseek-ai/cordis';
export declare const name = "govio-tools";
export declare const inject: string[];
/** 插件配置（来自 cordis.patch.yml 的 config 行，可被用户 profile patch 按行覆盖） */
export interface GovioToolsConfig {
    /** govio-cli 可执行命令或路径 */
    cliPath?: string;
    /** 子进程超时（毫秒） */
    timeoutMs?: number;
}
export declare function apply(ctx: Context, config: GovioToolsConfig): void;
