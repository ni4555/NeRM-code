def extract_prompt_from_response(response: str) -> str:
 """Extract the prompt from the response text that's bracketed with <prompt> and </prompt>"""
 try:
  # Find the last occurrence of <prompt> and </prompt>
  last_start = response.rfind("<prompt>")
  if last_start == -1:
   return response.strip()
  
  # Add length of <prompt> to get to the content start
  content_start = last_start + 8
  
  # Find the next </prompt> after the last <prompt>
  content_end = response.find("</prompt>", content_start)
  if content_end == -1:
   return response.strip()
  
  # Extract and return the content between the tags
  return response[content_start:content_end].strip()
 except Exception as e:
  print(e)
  return response.strip()


if __name__ == '__main__':
 prompt = ['Identifying the different parts between Prompt 1 and Prompt 2:\n\nPrompt 1: Design an efficient guided local search algorithm specifically tailored to optimize the Traveling Salesman Prob\
lem (TSP), aiming to determine the most cost-effective circular route that includes all specified nodes and returns to the starting point.\nPrompt 2: Develop an optimized guided local search approach t\
o tackle the classic optimization challenge of the Traveling Salesman Problem (TSP). This problem necessitates identifying the most efficient loop that encompasses all specified nodes and concludes at \
the origin point.\n\nDifferent parts:\n- "Design an efficient guided local search algorithm" vs "Develop an optimized guided local search approach"\n- "specifically tailored to optimize the Traveling S\
alesman Problem (TSP)" vs "to tackle the classic optimization challenge of the Traveling Salesman Problem (TSP)"\n- "aiming to determine the most cost-effective circular route" vs "necessitates identif\
ying the most efficient loop"\n- "that includes all specified nodes and returns to the starting point" vs "that encompasses all specified nodes and concludes at the origin point"\n\n2. Randomly mutate \
the different parts:\n- "Design an efficient guided local search algorithm" -> "Craft a highly effective guided local search heuristic"\n- "specifically tailored to optimize the Traveling Salesman Prob\
lem (TSP)" -> "customized for the Traveling Salesman Problem (TSP) optimization"\n- "aiming to determine the most cost-effective circular route" -> "with the goal of finding the least expensive circula\
r path"\n- "that includes all specified nodes and returns to the starting point" -> "which traverses all designated nodes and terminates at the initial node"\n\n3. Combine the different parts with Prom\
pt 3 and generate a final prompt bracketed with <prompt> and </prompt>:\n\nPrompt 3: Design a streamlined guided local search strategy to address the classic combinatorial optimization problem known as\
 the Traveling Salesman Problem (TSP). The objective is to pinpoint the most efficient circular route that traverses all designated nodes, ultimately returning to the starting point.\n\nFinal Prompt: <\
prompt>Craft a highly effective guided local search heuristic customized for the Traveling Salesman Problem (TSP) optimization. This heuristic should aim to find the least expensive circular path that \
traverses all designated nodes and terminates at the initial node, addressing the classic combinatorial optimization challenge.</prompt>']

 print(extract_prompt_from_response(response=prompt[0]))



