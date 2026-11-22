import { useCallback, useEffect } from "react";
import { useBeforeUnload, useBlocker } from "react-router-dom";

export function LandmarkLeaveWarning() {
  const blocker = useBlocker(true);

  useBeforeUnload(useCallback((event: BeforeUnloadEvent) => {
    event.preventDefault();
    event.returnValue = true;
  }, []));

  useEffect(() => {
    if (blocker.state !== "blocked") return;

    if (window.confirm("Leave this page? Your unexported landmarking changes will be lost.")) {
      blocker.proceed();
    } else {
      blocker.reset();
    }
  }, [blocker]);

  return null;
}
