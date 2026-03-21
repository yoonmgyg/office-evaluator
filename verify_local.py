import httpx
import json
import asyncio
from uuid import uuid4
from a2a.client import A2ACardResolver, ClientConfig, ClientFactory
from a2a.types import Message, Role, Part, TextPart, Task

async def verify_officeqa():
    evaluator_url = "http://127.0.0.1:9009"
    participant_url = "http://127.0.0.1:9019"

    print("Starting End-to-End Verification for OfficeQA...")
    
    async with httpx.AsyncClient(timeout=600.0) as httpx_client:
        resolver = A2ACardResolver(httpx_client=httpx_client, base_url=evaluator_url)
        try:
            agent_card = await resolver.get_agent_card()
            agent_card.url = evaluator_url
            print(f"Evaluator reachable: {agent_card.name}")
        except Exception as e:
            print(f"Evaluator not reachable: {e}")
            return

        config = ClientConfig(httpx_client=httpx_client)
        factory = ClientFactory(config)
        client = factory.create(agent_card)

        payload = {
            "participants": {"officeqa_agent": participant_url},
            "config": {
                "dataset_url": "https://raw.githubusercontent.com/databricks/officeqa/main/officeqa_full.csv",
                "num_questions": 1,
                "difficulty": "all",
                "tolerance": 0.02
            }
        }

        msg = Message(
            messageId=uuid4().hex,
            role=Role.user,
            parts=[Part(root=TextPart(kind="text", text=json.dumps(payload)))],
        )

        print("Sending evaluation request via A2A SDK...")
        
        task_id = None
        
        async for event in client.send_message(msg):
            if isinstance(event, tuple):
                task_obj, update = event
                if isinstance(task_obj, Task):
                    task_id = task_obj.id
                    print(f"Task {task_id} created. Monitoring...")
                    break
            elif isinstance(event, Task):
                task_id = event.id
                print(f"Task {task_id} created. Monitoring...")
                break

        if not task_id:
            print("No task was created by the evaluator.")
            return

        while True:
            task_status = await client.get_task({"id": task_id})
            state = task_status.status.state.value
            
            # Print latest update message
            status_msg = ""
            for p in task_status.status.message.parts:
                 if hasattr(p.root, "text"):
                     status_msg += p.root.text
            print(f"Status update: {status_msg}")
            
            if state in ["completed", "failed", "rejected", "canceled"]:
                print(f"Task finished with state: {state}")
                if state == "completed":
                    print("\nChecking artifacts for results:")
                    for artifact in task_status.artifacts:
                        if artifact.name == "evaluation_results":
                            for part in artifact.parts:
                                if hasattr(part.root, "data"):
                                    data = part.root.data
                                    print(f"Total Questions: {data.get('total_questions')}")
                                    print(f"Correct Answers: {data.get('correct_answers')}")
                                    print(f"Overall Accuracy: {data.get('accuracy')}")
                                    for r in data.get('results', []):
                                         print(f"\nQ: {r['question']}")
                                         print(f"GT: {r['ground_truth']}")
                                         print(f"Pred: {r['predicted']}")
                                         print(f"Correct: {r['is_correct']}")
                                         print(f"Trace: {r['reasoning_trace']}")
                break
            
            await asyncio.sleep(3)

if __name__ == "__main__":
    asyncio.run(verify_officeqa())
