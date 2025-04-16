from typing import Literal, Optional, List

from invokeai.app.invocations.baseinvocation import (
    BaseModel,
    BaseInvocation,
    BaseInvocationOutput,
    Classification,
    invocation,
    invocation_output,
)
from invokeai.app.invocations.fields import (
    InputField,
    OutputField,
    FluxUnoReferenceField
)
from invokeai.app.invocations.primitives import ImageField
from invokeai.app.services.shared.invocation_context import InvocationContext

@invocation_output("flux_redux_output")
class FluxUnoOutput(BaseInvocationOutput):
    """The conditioning output of a FLUX Redux invocation."""

    uno_refs: FluxUnoReferenceField = OutputField(
        description="Reference images container", title="Reference images"
    )

@invocation(
    "flux_redux",
    title="FLUX Redux",
    tags=["ip_adapter", "control"],
    category="ip_adapter",
    version="2.1.0",
    classification=Classification.Beta,
)
class FluxReduxInvocation(BaseInvocation):
    """Runs a FLUX Redux model to generate a conditioning tensor."""

    image: ImageField = InputField(description="The FLUX Redux image prompt.")
    image2: Optional[ImageField] = InputField(default=None, description="2nd reference")
    image3: Optional[ImageField] = InputField(default=None, description="3rd reference")
    image4: Optional[ImageField] = InputField(default=None, description="4th reference")

    def invoke(self, context: InvocationContext) -> FluxUnoOutput:
        images = [self.image.image_name]

        if self.image2 is not None:
            images.append(self.image2.image_name)
        if self.image3 is not None:
            images.append(self.image3.image_name)
        if self.image4 is not None:
            images.append(self.image4.image_name)
        
        return FluxUnoOutput(
            uno_refs=FluxUnoReferenceField(
                image_names=images)
        )
