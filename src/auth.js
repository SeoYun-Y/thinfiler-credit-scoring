// ------------------------------------------------------------------
// 하드코딩 계정 (데모/시연용 - 실제 인증 시스템 아님)
// ------------------------------------------------------------------
const ACCOUNTS = {
  trackA_user: {
    password: "trackA1234",
    role: "trackA",
    displayName: "트랙A 씬파일러",
    redirect: "track-a.html",
  },
  SYN235837: {
    password: "trackB1234",
    role: "trackB",
    displayName: "SYN_235837",
    redirect: "track-b.html",
  },
  reviewer: {
    password: "reviewer1234",
    role: "examiner",
    displayName: "심사자",
    redirect: "examiner.html",
  },
};

const SESSION_KEY = "creditDashboardSession";

function saveSession(userId, account) {
  sessionStorage.setItem(
    SESSION_KEY,
    JSON.stringify({ userId, role: account.role, displayName: account.displayName })
  );
}

function getSession() {
  try {
    return JSON.parse(sessionStorage.getItem(SESSION_KEY) || "null");
  } catch (e) {
    return null;
  }
}

function clearSession() {
  sessionStorage.removeItem(SESSION_KEY);
}

// 페이지 진입 시 로그인 여부 + 역할(role) 확인. 미로그인/역할불일치면 로그인 페이지로 이동.
function requireRole(expectedRole) {
  const session = getSession();
  if (!session || session.role !== expectedRole) {
    window.location.href = "index.html";
  }
  return session;
}

function logout() {
  clearSession();
  window.location.href = "index.html";
}
