# Leonardo HPC Access Guide

Credentials:
* username: `a08trc0e`
* psswd: `????`

```
ssh a08trc0e@login.leonardo.cineca.it
# you will be prompted for your password, type it and press Enter
``` 

The account for this school is `tra26_minwinsc` on the booster partition (`boost_usr_prod`).

Use `chappyner` to run jupyter notebooks on the compute nodes.

## SSH Host Key Mismatch Error

The address `login.leonardo.cineca.it` is actually a load-balancer that points to several different physical login nodes (like `login01`, `login02`, etc.). If CINECA recently updated their servers, or if you landed on a different login node than last time, your Mac notices that the "fingerprint" of the server doesn't match the one it saved previously. 

Open your local terminal and run this command to remove the old fingerprint:

```bash
ssh-keygen -R login.leonardo.cineca.it
```

*What this does:* It searches your `~/.ssh/known_hosts` file and automatically deletes the outdated entry for Leonardo.

### Try connecting again
Now, try to SSH into Leonardo again:

```bash
ssh <YOUR_USERNAME>@login.leonardo.cineca.it
```

It will give you a standard, much shorter warning saying:
> `The authenticity of host 'login.leonardo.cineca.it' can't be established... Are you sure you want to continue connecting (yes/no)?`

Type **`yes`** and press Enter. It will save the new key and prompt you for your password.

---

## Accessing Leonardo without typing a password every time

For security reasons, **CINECA does not allow standard static SSH keys** (like simply copying your `id_rsa.pub` into the `~/.ssh/authorized_keys` file) for external access to the Leonardo login nodes. If you try to do this from your local machine, the server will ignore the key and still prompt you for a password.

However, since typing a password every time you open a new terminal or transfer a file is frustrating, here are the best ways to get a "passwordless" and much better setup for your course.

### 1. The Best Workaround: SSH Multiplexing (Highly Recommended)
Since you can't use a standard key from the outside, the best trick is to use **SSH Multiplexing**. This tells your local SSH client to reuse an existing secure connection. You will only have to type your password **once**. Any subsequent terminal windows, `scp` file transfers, or VS Code sessions will log in instantly by piggybacking on the first connection.

**How to set it up (on your local Linux/Mac/WSL machine):**
1. Open your terminal and create a directory for the connection sockets:
   ```bash
   mkdir -p ~/.ssh/sockets
   ```
2. Edit or create the file `~/.ssh/config` on your local computer:
   ```bash
   nano ~/.ssh/config
   ```
3. Add the following block (replace `<YOUR_USERNAME>` with the training user you were assigned, e.g., `a08tra12`):
   ```text
   Host leonardo
       HostName login.leonardo.cineca.it
       User <YOUR_USERNAME>
       ControlMaster auto
       ControlPath ~/.ssh/sockets/%r@%h:%p
       ControlPersist 4h
   ```

**How to use it:**
Now, simply type `ssh leonardo`. It will ask for your password the first time. Leave that terminal running. If you open a second terminal and type `ssh leonardo`, you will bypass the password prompt entirely!

### 2. VS Code Remote - SSH
If you prefer a visual editor, VS Code's "Remote - SSH" extension works wonderfully with the multiplexing setup above. 
1. Install the **Remote - SSH** extension in VS Code.
2. Click the green `><` icon in the bottom left.
3. Select "Connect to Host" and choose `leonardo` (which it will read from your `~/.ssh/config` file).
4. It will ask for your password once to connect, and then you can edit files and use the integrated terminal seamlessly.

### 3. CINECA's Official Method: SmallStep Certificates 
*(Note: This is mandatory for permanent CINECA accounts, but temporary course accounts often skip this. Check with your instructor if your course user supports it).*

### 4. Internal SSH Keys (Login Node ➡️ Compute Node)
While you cannot use keys from your laptop *to* Leonardo, you **can and should** set up a key *inside* Leonardo. During the course, you may need to submit interactive jobs, run MPI codes, or tunnel into a compute node (e.g., to view a Jupyter Notebook). For this, the compute nodes need to trust your login node.

Once you have successfully logged into the Leonardo login node via password, run this exact sequence in the Leonardo terminal:
```bash
# Generate a key pair inside Leonardo (press Enter to leave the passphrase empty)
ssh-keygen -t ed25519 -N "" -f ~/.ssh/id_ed25519

# Authorize your own key
cat ~/.ssh/id_ed25519.pub >> ~/.ssh/authorized_keys
chmod 600 ~/.ssh/authorized_keys
```
This ensures that when your jobs run on the back-end GPU compute nodes, they can communicate with each other seamlessly without throwing permission errors.

---

## Stop known_hosts pollution in the future

SSH multiplexing only works as long as that initial "master" connection is alive (or for the 4 hours we set in `ControlPersist`). If you close your laptop, go to sleep, and try again tomorrow, the master connection will be dead. 

When you connect tomorrow, SSH will perform a brand-new handshake. It will hit the CINECA load balancer again, which might route you to a completely different login node (say, `login05` instead of `login02`). If `login05` has a different host key, SSH will either throw that scary error again or add a new line to your `known_hosts` file—eventually polluting it.

### How to stop cross-session pollution completely

Since you are taking a temporary course and Leonardo's load balancer rotates you among different servers, the cleanest way to handle this without messing up your Mac's main SSH setup is to **isolate Leonardo's keys into its own file**.

You can do this by updating your `~/.ssh/config` file. Open it (`nano ~/.ssh/config`) and make your `leonardo` block look exactly like this:

```text
Host leonardo
    HostName login.leonardo.cineca.it
    User <YOUR_USERNAME>
    
    # 1. Multiplexing (Keeps you on one node during the day)
    ControlMaster auto
    ControlPath ~/.ssh/sockets/%r@%h:%p
    ControlPersist 11h
    
    # 2. Prevent pollution and scary errors (From day to day)
    UserKnownHostsFile ~/.ssh/known_hosts_leonardo
    StrictHostKeyChecking no
    CheckHostIP no
```

Explanation:
1. **`UserKnownHostsFile ~/.ssh/known_hosts_leonardo`**: This tells SSH *not* to use your Mac's default `~/.ssh/known_hosts` file for this connection. Instead, it creates a separate mini-file just for CINECA. Your main `known_hosts` file stays perfectly clean.
2. **`StrictHostKeyChecking no`**: If you land on a new login node tomorrow, SSH will silently accept the new key and add it to the `known_hosts_leonardo` file without throwing an error. (actually it does not work, the error is still thrown, but it is enough to remove the `known_hosts_leonardo` file to fix it.)
3. **`CheckHostIP no`**: Since `login.leonardo.cineca.it` resolves to multiple IP addresses, this stops SSH from complaining when the IP address changes from yesterday's session.

With this setup, you get the best of both worlds: no typing passwords multiple times during the day, no scary warning screens tomorrow, and zero pollution in your main Mac security files!
