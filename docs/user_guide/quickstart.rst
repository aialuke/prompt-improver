Quick Start Guide
=================

Getting Started
--------------

1. **Start the MCP Server**

   .. code-block:: bash
   
      apes start --mcp-port 3000

2. **Run Health Checks**

   .. code-block:: bash
   
      apes health

3. **Process Your First Prompt**

   .. code-block:: python
   
      from prompt_improver import PromptImprovementService
      
      service = PromptImprovementService()
      result = await service.improve_prompt("Your prompt here")
      print(result)

Basic Configuration
------------------

Edit your `.env` file:

.. code-block:: bash

   POSTGRES_PASSWORD=your_secure_password
   REDIS_URL=redis://localhost:6379
   LOG_LEVEL=INFO
