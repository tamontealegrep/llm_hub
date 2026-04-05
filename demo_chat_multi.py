import asyncio

from app.bootstrap import build_container, pick_default_provider_and_model
from app.llm.contracts import StreamEventType


async def main() -> None:
    container = build_container()
    settings = container.settings
    registry = container.registry
    service = container.service

    provider_code, model_key = pick_default_provider_and_model(settings, registry)

    conversation = await service.create_conversation(
        provider_code=provider_code,
        model_key=model_key,
        system_prompt="Eres un asistente técnico, claro, preciso y útil.",
    )

    streaming_mode = True

    print("Conversación multiproveedor iniciada.")
    print(f"Providers disponibles: {', '.join(registry.available_provider_codes())}")
    print(f"Provider actual: {conversation.current_provider_code}")
    print(f"Modelo actual: {conversation.current_model_key}")
    print()
    print("Comandos:")
    print("  /switch <provider_code> <model_key>  -> cambia de proveedor y modelo")
    print("  /model <model_key>                   -> cambia solo el modelo actual")
    print("  /history                             -> muestra historial")
    print("  /stream on                           -> activa streaming")
    print("  /stream off                          -> desactiva streaming")
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
            print("\n--- HISTORIAL ---")
            print(f"Provider actual: {session.current_provider_code}")
            print(f"Modelo actual: {session.current_model_key}")
            for msg in session.messages:
                suffix = ""
                if msg.provider_code and msg.model_key:
                    suffix = f" [{msg.provider_code}:{msg.model_key}]"
                print(f"{msg.role.value.upper()}: {msg.content}{suffix}")
            print("-----------------\n")
            continue

        if user_input == "/stream on":
            streaming_mode = True
            print("Streaming activado.\n")
            continue

        if user_input == "/stream off":
            streaming_mode = False
            print("Streaming desactivado.\n")
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

            session = await service.change_model(
                conversation_id=conversation.id,
                provider_code=next_provider_code,
                model_key=next_model_key,
            )
            print(
                f"Cambiado a provider={session.current_provider_code}, model={session.current_model_key}\n"
            )
            continue

        if user_input.startswith("/model "):
            _, next_model_key = user_input.split(maxsplit=1)
            session = await service.change_model(
                conversation_id=conversation.id,
                model_key=next_model_key,
            )
            print(
                f"Modelo cambiado a provider={session.current_provider_code}, model={session.current_model_key}\n"
            )
            continue

        if streaming_mode:
            stream = await service.stream_message(
                conversation_id=conversation.id,
                user_text=user_input,
            )

            print("Asistente: ", end="", flush=True)

            async for event in stream:
                if event.type == StreamEventType.DELTA and event.delta:
                    print(event.delta, end="", flush=True)
                elif event.type == StreamEventType.ERROR:
                    print(f"\n[ERROR] {event.error_code}: {event.error_message}")
                    break
                elif event.type == StreamEventType.END:
                    print("\n")
        else:
            result = await service.send_message(
                conversation_id=conversation.id,
                user_text=user_input,
            )
            print(f"Asistente: {result.content}\n")


if __name__ == "__main__":
    asyncio.run(main())