# AWS Changes

What changed in the AWS setup (region `ap-southeast-2`, account `771904882759`) and why.
No secret values are stored in this file.

---

## 1. App secrets moved to Secrets Manager

**What:** These values from `.env` are now stored in AWS Secrets Manager, all named with the `multitenant/` prefix:

| Secret | Used for |
|---|---|
| `multitenant/DATABASE_URL` | Postgres connection (already existed) |
| `multitenant/GROQ_API_KEY` | Groq LLM calls |
| `multitenant/GEMINI_API_KEY` | Gemini LLM calls |
| `multitenant/LANGSMITH_API_KEY` | LangSmith tracing |
| `multitenant/JWT_SECRET_<TENANT>` (4 secrets) | Verifying each tenant's customer JWTs |

**Why:** The container gets these values from AWS when it starts, so they aren't baked into the image or written into the task definition.
`AGENT_MODE` and `APP_ENV` aren't secrets. The Dockerfile sets them.

---

## 2. ECS task execution role can read the new secrets

**What:** The policy on `ecsTaskExecutionRole` now allows `secretsmanager:GetSecretValue` on
`arn:aws:secretsmanager:ap-southeast-2:771904882759:secret:multitenant/*`.

**Why:** Task definition revision 4 references the new secrets, but the role could only read `DATABASE_URL`.
Every task failed to start with `AccessDenied`, and ECS kept retrying, so revision 4 never ran.
With the wildcard, any future `multitenant/...` secret works without another IAM change.

> This is the *execution* role. ECS uses it only to fetch the secrets at startup. App code inside the container can't use it to read Secrets Manager.

---

## 3. Database migrations run automatically on every deploy

**Problem found:** After revision 4 started, `/health` returned 200 but `/agent` returned **500** with
`relation "tenants" does not exist`. The production database had no tables: the image includes the Alembic migrations, but nothing ever ran them.

**What changed:** `.github/workflows/cd.yml` now does this on every push to `master`:

1. Build and push the image to ECR (as before).
2. Register a new task definition revision. It copies the latest revision in AWS and only changes the image to `:<commit-sha>`.
3. Run a one-off ECS task with that revision and the command `alembic upgrade head`.
   **If migrations fail, the deploy stops**, and the old version keeps serving.
4. Update the ECS service to the new revision and wait until it is stable.

**Why this approach:**
- Migrations run once per deploy, not once per container, so scaling to many tasks is safe.
- A broken migration never takes the live app down.
- A `concurrency` lock stops two deploys from migrating the database at the same time.
- Task settings (CPU, memory, secrets, roles) are still managed in the AWS console, and CD carries them forward.

**Rule for future migrations:** keep them backward-compatible. They run while the old version is still serving, so add the new column in one deploy and drop the old one in a later deploy.

---

## 4. GitHub deploy user: updated IAM policy

The IAM user whose keys are in the GitHub secrets (`AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY`) could previously only push to ECR.
It now also needs permission to register task definitions, run the migration task and update the service.
Replace its policy with this one:


| Statement | Why it's needed |
|---|---|
| `ECRAuthorization`, `PushToSpecificRepository` | Unchanged: log in to ECR and push the image |
| `TaskDefinitions` | Read the current revision and register a new one with the new image. These actions can't be limited to a single resource in AWS, hence `*` |
| `RunMigrationTask` | Start the `alembic upgrade head` task, only from the `multitenant-api` family and only in our cluster |
| `WatchMigrationTask` | Wait for the migration task and read its exit code |
| `DeployService` | Read the service's network settings and switch it to the new revision |
| `ReadMigrationLogs` | Print the migration output in the GitHub Actions log |
| `PassTaskRolesToEcs` | Registering a task definition or running a task hands its role to ECS, which requires `PassRole` |

> If the task definition has a **task role** (`taskRoleArn`) as well, add that role's ARN to the `PassTaskRolesToEcs` resource.

---

## 5. Still to do (one-time, manual)

- **Seed production data.** The first deploy creates the tables, but they start empty. Run a one-off ECS task, once, to create the
  `megamart` and `tiny-candles` tenants with their fixed UUIDs (the JWT secret names depend on them) and seed customers and orders.
  This isn't in CD on purpose: it should run once, not on every deploy.
- **Production API keys.** Create new keys for each tenant and store them as `multitenant/MEGAMART_API_KEY` and
  `multitenant/TINY_CANDLES_API_KEY`. Don't use the dev placeholder keys from the frontend in production.
- **Knowledge base.** Policy and RAG questions need documents loaded (`scripts/ingest_megamart_kb.py`). That script isn't in the image yet.
