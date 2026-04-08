#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
demo.py — Demo interactivo unificado del LLM Hub.

Reemplaza: demo_chat.py, demo_chat_multi.py, demo_chat_stream.py, demo_chat_tools.py

Características:
  - Streaming y modo blocking seleccionables en runtime
  - Tool calling funcional en ambos modos
  - Cambio de provider y modelo en caliente
  - Historial enriquecido (muestra tool_calls y resultados)
  - Diagnóstico en vivo de eventos de stream

Uso:
  python demo.py
  python demo.py --no-stream        # inicia en modo blocking
  python demo.py --tools none       # inicia sin tools activas
  python demo.py --provider anthropic --model claude-3-5-sonnet-latest
"""

import argparse
import asyncio
import json
import sys
import textwrap

from app.bootstrap import build_container, pick_default_chat_provider_and_model
from app.shared.exceptions import AppError
from app.capabilities.chat.contracts import ChatStreamEventType

# ──────────────────────────────────────────────────────────────────────────────
# Colores ANSI (se deshabilitan si stdout no es terminal)
# ──────────────────────────────────────────────────────────────────────────────

_USE_COLOR = sys.stdout.isatty()


def _c(code: str, text: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _USE_COLOR else text


def gray(t: str) -> str:    return _c("90", t)
def green(t: str) -> str:   return _c("32", t)
def yellow(t: str) -> str:  return _c("33", t)
def cyan(t: str) -> str:    return _c("36", t)
def red(t: str) -> str:     return _c("31", t)
def bold(t: str) -> str:    return _c("1", t)
def dim(t: str) -> str:     return _c("2", t)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers de presentación
# ──────────────────────────────────────────────────────────────────────────────

def _json_dump(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2)


def _parse_tool_names(raw: str) -> list[str]:
    return [p.strip() for p in raw.replace(",", " ").split() if p.strip()]


def _print_separator(char: str = "─", width: int = 60) -> None:
    print(dim(char * width))


def _print_history(session) -> None:
    _print_separator()
    print(bold("  HISTORIAL DE CONVERSACIÓN"))
    print(dim(f"  ID          : {session.id}"))
    print(dim(f"  Provider    : {session.current_provider_code}"))
    print(dim(f"  Modelo      : {session.current_model_key}"))
    print(dim(f"  Mensajes    : {len(session.messages)}"))
    _print_separator()

    for i, msg in enumerate(session.messages, 1):
        role = msg.role.value
        provider_tag = ""
        if msg.provider_code and msg.model_key:
            provider_tag = dim(f" [{msg.provider_code}:{msg.model_key}]")

        if role == "user":
            print(f"\n  {bold(cyan(f'[{i}] USER'))}{provider_tag}")
            print(f"  {msg.content}")

        elif role == "assistant":
            print(f"\n  {bold(green(f'[{i}] ASSISTANT'))}{provider_tag}")
            content_display = msg.content if msg.content else dim("(sin texto — solo tool_calls)")
            print(f"  {content_display}")

            if msg.tool_calls:
                for tc in msg.tool_calls:
                    print(f"  {yellow('→ TOOL_CALL')} {dim(f'id={tc.id}')}")
                    print(f"    name      : {bold(tc.name)}")
                    args_str = json.dumps(tc.arguments, ensure_ascii=False)
                    print(f"    arguments : {args_str}")

        elif role == "tool":
            tool_label = msg.name or "(sin nombre)"
            print(f"\n  {bold(yellow(f'[{i}] TOOL'))} {dim(f'← {tool_label}')}")
            print(f"  {dim(f'tool_call_id: {msg.tool_call_id}')}")
            # Intentar mostrar el contenido como JSON formateado
            try:
                parsed = json.loads(msg.content)
                pretty = json.dumps(parsed, ensure_ascii=False, indent=2)
                for line in pretty.splitlines():
                    print(f"  {line}")
            except (json.JSONDecodeError, TypeError):
                print(f"  {msg.content}")

        else:
            print(f"\n  {bold(f'[{i}] {role.upper()}')}{provider_tag}")
            print(f"  {msg.content}")

    _print_separator()
    print()


def _print_tools_status(
    *,
    available_tool_names: list[str],
    enabled_tool_names: list[str],
) -> None:
    _print_separator()
    print(bold("  TOOLS"))
    avail_str = ", ".join(available_tool_names) if available_tool_names else dim("(ninguna)")
    enabl_str = ", ".join(enabled_tool_names) if enabled_tool_names else dim("(ninguna)")
    print(f"  Disponibles : {avail_str}")
    print(f"  Activas     : {green(enabl_str) if enabled_tool_names else dim('(ninguna)')}")
    _print_separator()
    print()


def _print_help() -> None:
    help_text = """
  COMANDOS DISPONIBLES
  ────────────────────────────────────────────────────────────
  Conversación:
    /stream on|off              Activa o desactiva el streaming
    /history                    Muestra el historial completo

  Provider y modelo:
    /switch <provider> <model>  Cambia provider y modelo
    /model <model>              Cambia solo el modelo
    /providers                  Lista los providers disponibles

  Tools:
    /tools list                 Muestra tools disponibles y activas
    /tools all                  Activa todas las tools
    /tools none                 Desactiva todas las tools
    /tools use <t1> [t2 ...]    Activa solo esas tools

  General:
    /help                       Muestra esta ayuda
    /status                     Estado actual del demo
    /exit                       Salir

  PROVEEDORES DISPONIBLES (códigos)
    openai · anthropic · google · xai
  ────────────────────────────────────────────────────────────
"""
    print(help_text)


def _print_status(
    *,
    session,
    streaming_mode: bool,
    enabled_tool_names: list[str],
    available_tool_names: list[str],
) -> None:
    _print_separator()
    print(bold("  ESTADO ACTUAL"))
    print(f"  Provider    : {bold(session.current_provider_code)}")
    print(f"  Modelo      : {bold(session.current_model_key)}")
    mode_str = green("STREAMING") if streaming_mode else yellow("BLOCKING")
    print(f"  Modo        : {mode_str}")
    tools_str = green(", ".join(enabled_tool_names)) if enabled_tool_names else dim("(ninguna)")
    print(f"  Tools activas: {tools_str}")
    print(f"  Tools disponibles: {', '.join(available_tool_names) or dim('(ninguna)')}")
    print(f"  Mensajes    : {len(session.messages)}")
    _print_separator()
    print()


# ──────────────────────────────────────────────────────────────────────────────
# Lógica de send_message (blocking)
# ──────────────────────────────────────────────────────────────────────────────

async def _handle_blocking(
    *,
    service,
    conversation,
    user_input: str,
    enabled_tool_names: list[str],
) -> None:
    """Invocación blocking con tool calling completo."""
    # Pasamos lista explícita vacía si no hay tools activas,
    # o la lista de tools activas, nunca None (que usaría todas).
    tool_names_arg = enabled_tool_names if enabled_tool_names else []

    result = await service.send_message(
        conversation_id=conversation.id,
        user_text=user_input,
        tool_names=tool_names_arg,
    )

    print(f"\n{bold(green('Asistente:'))} ", end="")
    content = result.content if result.content else dim("(sin texto en la respuesta final)")
    print(content)

    if result.tool_calls:
        print(dim(f"\n  [Tool calls en la respuesta final: {len(result.tool_calls)}]"))

    if result.usage:
        u = result.usage
        print(dim(
            f"  [tokens: prompt={u.prompt_tokens} "
            f"completion={u.completion_tokens} "
            f"total={u.total_tokens}]"
        ))

    print()


# ──────────────────────────────────────────────────────────────────────────────
# Lógica de stream_message (streaming)
# ──────────────────────────────────────────────────────────────────────────────

async def _handle_streaming(
    *,
    service,
    conversation,
    user_input: str,
    enabled_tool_names: list[str],
) -> None:
    """Streaming con tool calling completo y visualización enriquecida."""
    tool_names_arg = enabled_tool_names if enabled_tool_names else []

    stream = await service.stream_message(
        conversation_id=conversation.id,
        user_text=user_input,
        tool_names=tool_names_arg,
    )

    in_assistant_turn = False
    tool_iteration = 0

    async for event in stream:

        if event.type == ChatStreamEventType.START:
            if not in_assistant_turn:
                print(f"\n{bold(green('Asistente:'))} ", end="", flush=True)
                in_assistant_turn = True
            else:
                # Turno subsiguiente tras ejecutar tools
                tool_iteration += 1
                print(f"\n{bold(green('Asistente:'))} {dim(f'(tras tools, iter {tool_iteration})')}", end=" ", flush=True)

        elif event.type == ChatStreamEventType.DELTA:
            if event.delta:
                print(event.delta, end="", flush=True)

        elif event.type == ChatStreamEventType.TOOL_USE:
            # El servicio ya ejecutó las tools; aquí solo mostramos el resumen
            print()  # nueva línea tras el texto que hubiera
            calls = event.tool_calls
            executed = (event.raw_event or {}).get("executed_tools", [])

            print(dim(f"\n  ┌─ Tool calling ({len(calls)} call{'s' if len(calls) != 1 else ''}) ─────────────"))
            for i, tc in enumerate(calls):
                print(dim(f"  │ [{i+1}] {bold(tc.name)}"))
                args_str = json.dumps(tc.arguments, ensure_ascii=False)
                print(dim(f"  │     args    : {args_str}"))
                # Buscar el resultado ejecutado correspondiente
                exec_info = next((e for e in executed if e.get("tool_call_id") == tc.id), None)
                if exec_info:
                    result_preview = exec_info.get("content", "")
                    if len(result_preview) > 120:
                        result_preview = result_preview[:120] + "…"
                    status = red("ERROR") if exec_info.get("is_error") else green("OK")
                    print(dim(f"  │     resultado: [{status}{dim(']')} {result_preview}"))
            print(dim(f"  └{'─' * 48}"))
            in_assistant_turn = False  # el próximo START es un nuevo turno

        elif event.type == ChatStreamEventType.END:
            print("\n")  # línea limpia al terminar

        elif event.type == ChatStreamEventType.ERROR:
            print(f"\n{red(f'[ERROR] {event.error_code}: {event.error_message}')}\n")
            return


# ──────────────────────────────────────────────────────────────────────────────
# Loop principal
# ──────────────────────────────────────────────────────────────────────────────

async def main(args: argparse.Namespace) -> None:
    container = build_container(with_builtin_tools=True)
    settings  = container.settings
    registry  = container.provider_registry
    tool_registry = container.tool_registry
    service   = container.chat_service

    # Determinar provider y modelo iniciales
    if args.provider and args.model:
        if not registry.has(args.provider):
            print(red(f"[ERROR] Provider '{args.provider}' no está disponible."))
            print(f"Disponibles: {', '.join(registry.available_provider_codes())}")
            sys.exit(1)
        provider_code = args.provider
        model_key = args.model
    elif args.provider:
        print(red("[ERROR] Si especificas --provider debes también especificar --model."))
        sys.exit(1)
    else:
        provider_code, model_key = pick_default_chat_provider_and_model(settings, registry)

    conversation = await service.create_conversation(
        provider_code=provider_code,
        model_key=model_key,
        system_prompt=(
            "Eres un asistente técnico, claro, preciso y útil. "
            "Cuando una herramienta disponible te permita responder mejor o "
            "con datos más precisos, úsala. Si no hace falta, responde normalmente. "
            "Después de usar una herramienta, integra el resultado en una respuesta final."
        ),
    )

    available_tool_names = tool_registry.available_tool_names()

    # Modo de streaming inicial
    streaming_mode: bool = not args.no_stream

    # Tools activas iniciales
    if args.tools == "none":
        enabled_tool_names: list[str] = []
    elif args.tools == "all" or args.tools is None:
        enabled_tool_names = list(available_tool_names)
    else:
        # nombres específicos separados por coma
        enabled_tool_names = _parse_tool_names(args.tools)

    # ── Banner de bienvenida ──────────────────────────────────────────────────
    print()
    _print_separator("═")
    print(bold("  LLM HUB — Demo interactivo unificado"))
    _print_separator("═")
    print(f"  Providers disponibles : {', '.join(registry.available_provider_codes())}")
    print(f"  Provider actual       : {bold(conversation.current_provider_code)}")
    print(f"  Modelo actual         : {bold(conversation.current_model_key)}")
    mode_str = green("STREAMING") if streaming_mode else yellow("BLOCKING")
    print(f"  Modo inicial          : {mode_str}")
    tools_str = green(", ".join(enabled_tool_names)) if enabled_tool_names else dim("(ninguna)")
    print(f"  Tools activas         : {tools_str}")
    print()
    print(dim("  Escribe /help para ver los comandos disponibles."))
    _print_separator("═")
    print()

    # ── Loop de conversación ──────────────────────────────────────────────────
    while True:
        try:
            prompt_provider = dim(f"[{conversation.current_provider_code}:{conversation.current_model_key}]")
            mode_indicator = green("~") if streaming_mode else yellow("■")
            raw = input(f"{mode_indicator} {prompt_provider} {bold('Tú:')} ").strip()
        except (EOFError, KeyboardInterrupt):
            print(f"\n{dim('Saliendo...')}\n")
            break

        if not raw:
            continue

        # ── Comandos del sistema ──────────────────────────────────────────────

        if raw == "/exit":
            print(dim("\nHasta luego.\n"))
            break

        if raw == "/help":
            _print_help()
            continue

        if raw == "/history":
            session = await service.get_conversation(conversation.id)
            _print_history(session)
            continue

        if raw == "/providers":
            print(f"\n  Disponibles: {', '.join(registry.available_provider_codes())}\n")
            continue

        if raw == "/status":
            session = await service.get_conversation(conversation.id)
            _print_status(
                session=session,
                streaming_mode=streaming_mode,
                enabled_tool_names=enabled_tool_names,
                available_tool_names=available_tool_names,
            )
            continue

        if raw in ("/stream on", "/stream off"):
            streaming_mode = raw == "/stream on"
            mode_str = green("STREAMING") if streaming_mode else yellow("BLOCKING")
            print(f"\n  Modo cambiado a {mode_str}\n")
            continue

        if raw == "/tools list":
            _print_tools_status(
                available_tool_names=available_tool_names,
                enabled_tool_names=enabled_tool_names,
            )
            continue

        if raw == "/tools all":
            enabled_tool_names = list(available_tool_names)
            print(f"\n  {green('Tools activadas:')} {', '.join(enabled_tool_names) or dim('(ninguna)')}\n")
            continue

        if raw == "/tools none":
            enabled_tool_names = []
            print(f"\n  {yellow('Todas las tools desactivadas.')}\n")
            continue

        if raw.startswith("/tools use "):
            selected = _parse_tool_names(raw[len("/tools use "):].strip())
            if not selected:
                print(f"\n  {red('Uso:')} /tools use <tool1> [tool2 ...]\n")
                continue
            invalid = [n for n in selected if not tool_registry.has(n)]
            if invalid:
                print(f"\n  {red('Tools no disponibles:')} {', '.join(invalid)}\n")
                continue
            # Deduplicar manteniendo orden
            seen: set[str] = set()
            deduped: list[str] = []
            for n in selected:
                if n not in seen:
                    deduped.append(n)
                    seen.add(n)
            enabled_tool_names = deduped
            print(f"\n  {green('Tools activas:')} {', '.join(enabled_tool_names)}\n")
            continue

        if raw.startswith("/switch "):
            parts = raw.split(maxsplit=2)
            if len(parts) != 3:
                print(f"\n  {red('Uso:')} /switch <provider_code> <model_key>\n")
                continue
            _, next_provider, next_model = parts
            if not registry.has(next_provider):
                print(f"\n  {red(f'Provider no disponible: {next_provider}')}\n")
                continue
            try:
                session = await service.change_model(
                    conversation_id=conversation.id,
                    provider_code=next_provider,
                    model_key=next_model,
                )
                conversation = session
                print(f"\n  {green('Cambiado a')} provider={bold(next_provider)} modelo={bold(next_model)}\n")
            except AppError as exc:
                print(f"\n  {red(f'[ERROR] {exc.error_code}: {exc.message}')}\n")
            continue

        if raw.startswith("/model "):
            parts = raw.split(maxsplit=1)
            if len(parts) != 2:
                print(f"\n  {red('Uso:')} /model <model_key>\n")
                continue
            next_model = parts[1].strip()
            try:
                session = await service.change_model(
                    conversation_id=conversation.id,
                    model_key=next_model,
                )
                conversation = session
                print(f"\n  {green('Modelo cambiado a')} {bold(next_model)}\n")
            except AppError as exc:
                print(f"\n  {red(f'[ERROR] {exc.error_code}: {exc.message}')}\n")
            continue

        if raw.startswith("/"):
            print(f"\n  {red(f'Comando desconocido: {raw}')}")
            print(dim("  Escribe /help para ver los comandos disponibles.\n"))
            continue

        # ── Mensaje al LLM ────────────────────────────────────────────────────
        try:
            if streaming_mode:
                await _handle_streaming(
                    service=service,
                    conversation=conversation,
                    user_input=raw,
                    enabled_tool_names=enabled_tool_names,
                )
            else:
                await _handle_blocking(
                    service=service,
                    conversation=conversation,
                    user_input=raw,
                    enabled_tool_names=enabled_tool_names,
                )

        except AppError as exc:
            print(f"\n{red(f'[ERROR] {exc.error_code}: {exc.message}')}\n")

        except Exception as exc:
            print(f"\n{red(f'[ERROR] {exc.__class__.__name__}: {exc}')}\n")

        # Refrescar la referencia a la conversación para tener el estado actual
        conversation = await service.get_conversation(conversation.id)


# ──────────────────────────────────────────────────────────────────────────────
# Entrypoint
# ──────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="LLM Hub — demo interactivo unificado",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Ejemplos:
              python demo.py
              python demo.py --no-stream
              python demo.py --tools none
              python demo.py --provider anthropic --model claude-3-5-sonnet-latest
              python demo.py --provider openai --model gpt-4o --tools "get_current_utc_time,sum_numbers"
        """),
    )
    parser.add_argument(
        "--no-stream",
        action="store_true",
        default=False,
        help="Inicia en modo blocking (sin streaming). Default: streaming activado.",
    )
    parser.add_argument(
        "--tools",
        default=None,
        metavar="NAMES|all|none",
        help=(
            "Tools a activar al inicio. "
            "'all' activa todas (default), 'none' desactiva todas, "
            "o una lista separada por comas: 'tool1,tool2'."
        ),
    )
    parser.add_argument(
        "--provider",
        default=None,
        metavar="CODE",
        help="Provider inicial (openai, anthropic, google, xai).",
    )
    parser.add_argument(
        "--model",
        default=None,
        metavar="MODEL_KEY",
        help="Modelo inicial. Requerido si se especifica --provider.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    asyncio.run(main(_parse_args()))