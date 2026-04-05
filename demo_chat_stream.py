import asyncio
import json

from app.bootstrap import build_container, pick_default_provider_and_model
from app.core.exceptions import AppError
from app.llm.contracts import StreamEventType


def _json_dump(value: object) -> str:
    return json.dumps(value, ensure_ascii=False)


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
        system_prompt="Eres un asistente técnico, claro, preciso y útil.",
    )

    available_tool_names = tool_registry.available_tool_names()

    print("Conversación multiproveedor iniciada (modo streaming).")
    print(f"Providers disponibles: {', '.join(registry.available_provider_codes())}")
    print(f"Provider actual: {conversation.current_provider_code}")
    print(f"Modelo actual: {conversation.current_model_key}")
    print(
        f"Tools disponibles (informativo): {', '.join(available_tool_names) if available_tool_names else '(ninguna)'}"
    )
    print()
    print(
        "Nota: en Fase 1 este demo NO usa tools, porque stream_message() "
        "todavía no soporta tool calling."
    )
    print("Si quieres probar tools usa demo_chat.py o demo_chat_tools.py.")
    print()
    print("Comandos:")
    print("  /switch <provider_code> <model_key>  -> cambia de proveedor y modelo")
    print("  /model <model_key>                   -> cambia solo el modelo actual")
    print("  /history                             -> muestra historial")
    print("  /tools list                          -> muestra tools disponibles (informativo)")
    print("  /exit                                -> salir")
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

        if user_input == "/tools list":
            print(
                "\nTools disponibles (informativo): "
                f"{', '.join(available_tool_names) if available_tool_names else '(ninguna)'}"
            )
            print(
                "Este demo no las usa porque el tool calling en streaming "
                "no está implementado en la Fase 1.\n"
            )
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


if __name__ == "__main__":
    asyncio.run(main())