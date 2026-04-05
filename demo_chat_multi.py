import asyncio
import json

from app.bootstrap import build_container, pick_default_provider_and_model
from app.core.exceptions import AppError
from app.llm.contracts import StreamEventType


def _json_dump(value: object) -> str:
    return json.dumps(value, ensure_ascii=False)


def _parse_tool_names(raw: str) -> list[str]:
    return [part.strip() for part in raw.replace(",", " ").split() if part.strip()]


def _print_history(session) -> None:
    print("\n--- HISTORIAL ---")
    print(f"Provider actual: {session.current_provider_code}")
    print(f"Modelo actual: {session.current_model_key}")

    for msg in session.messages:
        suffix = ""
        if msg.provider_code and msg.model_key:
            suffix = f" [{msg.provider_code}:{msg.model_key}]"

        if msg.role.value == "assistant":
            content = msg.content if msg.content else "(sin contenido textual)"
            print(f"ASSISTANT: {content}{suffix}")

            if msg.tool_calls:
                for tool_call in msg.tool_calls:
                    print(
                        "    -> TOOL_CALL "
                        f"id={tool_call.id} "
                        f"name={tool_call.name} "
                        f"args={_json_dump(tool_call.arguments)}"
                    )

        elif msg.role.value == "tool":
            tool_label = msg.name or "(sin nombre)"
            print(
                f"TOOL: {msg.content} "
                f"[tool_call_id={msg.tool_call_id}, name={tool_label}]"
            )

        else:
            print(f"{msg.role.value.upper()}: {msg.content}{suffix}")

    print("-----------------\n")


def _print_tools_status(
    *,
    available_tool_names: list[str],
    enabled_tool_names: list[str],
) -> None:
    print("\n--- TOOLS ---")
    print(
        f"Disponibles: {', '.join(available_tool_names) if available_tool_names else '(ninguna)'}"
    )
    print(
        f"Activas: {', '.join(enabled_tool_names) if enabled_tool_names else '(ninguna)'}"
    )
    print("-------------\n")


async def main() -> None:
    container = build_container(with_builtin_tools=True)
    settings = container.settings
    registry = container.registry
    tool_registry = container.tool_registry
    service = container.service

    provider_code, model_key = pick_default_provider_and_model(settings, registry)

    conversation = await service.create_conversation(
        provider_code=provider_code,
        model_key=model_key,
        system_prompt=(
            "Eres un asistente técnico, claro, preciso y útil. "
            "Si se te proporcionan herramientas y son necesarias para responder mejor, úsalas. "
            "Si no hacen falta, responde normalmente."
        ),
    )

    streaming_mode = True
    available_tool_names = tool_registry.available_tool_names()
    enabled_tool_names: list[str] = []

    print("Conversación multiproveedor iniciada.")
    print(f"Providers disponibles: {', '.join(registry.available_provider_codes())}")
    print(f"Provider actual: {conversation.current_provider_code}")
    print(f"Modelo actual: {conversation.current_model_key}")
    print(
        f"Tools disponibles: {', '.join(available_tool_names) if available_tool_names else '(ninguna)'}"
    )
    print("Tools activas al inicio: (ninguna)")
    print()
    print("Nota: en Fase 1 las tools solo funcionan con /stream off.")
    print()
    print("Comandos:")
    print("  /switch <provider_code> <model_key>   -> cambia de proveedor y modelo")
    print("  /model <model_key>                    -> cambia solo el modelo actual")
    print("  /history                              -> muestra historial")
    print("  /stream on                            -> activa streaming")
    print("  /stream off                           -> desactiva streaming")
    print("  /tools list                           -> muestra tools disponibles y activas")
    print("  /tools all                            -> activa todas las tools")
    print("  /tools none                           -> desactiva todas las tools")
    print("  /tools use <tool1> [tool2 ...]        -> activa solo esas tools")
    print("  /exit                                 -> salir")
    print()

    while True:
        user_input = input("Tú: ").strip()

        if not user_input:
            continue

        if user_input == "/exit":
            break

        if user_input == "/history":
            session = await service.get_conversation(conversation.id)
            _print_history(session)
            continue

        if user_input == "/stream on":
            streaming_mode = True
            if enabled_tool_names:
                print(
                    "Streaming activado. Recuerda: en esta fase las tools no funcionan "
                    "con streaming. Usa /stream off o /tools none antes de enviar mensajes.\n"
                )
            else:
                print("Streaming activado.\n")
            continue

        if user_input == "/stream off":
            streaming_mode = False
            print("Streaming desactivado.\n")
            continue

        if user_input == "/tools list":
            _print_tools_status(
                available_tool_names=available_tool_names,
                enabled_tool_names=enabled_tool_names,
            )
            continue

        if user_input == "/tools all":
            enabled_tool_names = list(available_tool_names)
            print(
                f"Tools activas: {', '.join(enabled_tool_names) if enabled_tool_names else '(ninguna)'}\n"
            )
            continue

        if user_input == "/tools none":
            enabled_tool_names = []
            print("Todas las tools fueron desactivadas.\n")
            continue

        if user_input.startswith("/tools use "):
            raw = user_input[len("/tools use ") :].strip()
            selected_tool_names = _parse_tool_names(raw)

            if not selected_tool_names:
                print("Uso: /tools use <tool1> [tool2 ...]\n")
                continue

            invalid_tool_names = [
                tool_name
                for tool_name in selected_tool_names
                if not tool_registry.has(tool_name)
            ]
            if invalid_tool_names:
                print(f"Tools no disponibles: {', '.join(invalid_tool_names)}\n")
                continue

            deduped: list[str] = []
            seen: set[str] = set()
            for tool_name in selected_tool_names:
                if tool_name not in seen:
                    deduped.append(tool_name)
                    seen.add(tool_name)

            enabled_tool_names = deduped
            print(f"Tools activas: {', '.join(enabled_tool_names)}\n")
            continue

        if user_input.startswith("/switch "):
            parts = user_input.split(maxsplit=2)
            if len(parts) != 3:
                print("Uso: /switch <provider_code> <model_key>\n")
                continue

            _, next_provider_code, next_model_key = parts

            if not registry.has(next_provider_code):
                print(f"Provider no disponible: {next_provider_code}\n")
                continue

            try:
                session = await service.change_model(
                    conversation_id=conversation.id,
                    provider_code=next_provider_code,
                    model_key=next_model_key,
                )
            except AppError as exc:
                print(f"[ERROR] {exc.error_code}: {exc.message}\n")
                continue

            print(
                f"Cambiado a provider={session.current_provider_code}, "
                f"model={session.current_model_key}\n"
            )
            continue

        if user_input.startswith("/model "):
            _, next_model_key = user_input.split(maxsplit=1)

            try:
                session = await service.change_model(
                    conversation_id=conversation.id,
                    model_key=next_model_key,
                )
            except AppError as exc:
                print(f"[ERROR] {exc.error_code}: {exc.message}\n")
                continue

            print(
                f"Modelo cambiado a provider={session.current_provider_code}, "
                f"model={session.current_model_key}\n"
            )
            continue

        if streaming_mode:
            if enabled_tool_names:
                print(
                    "Las tools no están soportadas en streaming durante la fase 1. "
                    "Usa /stream off o /tools none.\n"
                )
                continue

            try:
                stream = await service.stream_message(
                    conversation_id=conversation.id,
                    user_text=user_input,
                )

                print("Asistente: ", end="", flush=True)

                async for event in stream:
                    if event.type == StreamEventType.DELTA and event.delta:
                        print(event.delta, end="", flush=True)
                    elif event.type == StreamEventType.ERROR:
                        print(f"\n[ERROR] {event.error_code}: {event.error_message}\n")
                        break
                    elif event.type == StreamEventType.END:
                        print("\n")

            except AppError as exc:
                print(f"\n[ERROR] {exc.error_code}: {exc.message}\n")

            except Exception as exc:
                print(f"\n[ERROR] {exc.__class__.__name__}: {exc}\n")

        else:
            try:
                result = await service.send_message(
                    conversation_id=conversation.id,
                    user_text=user_input,
                    tool_names=enabled_tool_names,
                )
                assistant_text = result.content if result.content else "(sin contenido textual)"
                print(f"Asistente: {assistant_text}\n")

            except AppError as exc:
                print(f"[ERROR] {exc.error_code}: {exc.message}\n")

            except Exception as exc:
                print(f"[ERROR] {exc.__class__.__name__}: {exc}\n")


if __name__ == "__main__":
    asyncio.run(main())