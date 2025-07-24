Installation Guide
==================

System Requirements
------------------

* Python 3.11 or higher
* PostgreSQL 13+ (for database operations)
* Redis 6+ (for caching)
* 4GB RAM minimum, 8GB recommended

Quick Installation
-----------------

.. code-block:: bash

   # Clone the repository
   git clone https://github.com/your-org/apes.git
   cd apes
   
   # Install dependencies
   pip install -e .
   
   # Set up environment
   cp .env.example .env
   # Edit .env with your configuration
   
   # Initialize database
   apes db init
   
   # Start services
   apes start

Docker Installation
------------------

.. code-block:: bash

   # Using Docker Compose
   docker-compose up -d
   
   # Verify installation
   docker-compose exec apes apes health
